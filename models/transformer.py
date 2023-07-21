from einops import rearrange, repeat
import torch
from torch import nn
F = nn.functional

from .implicit_transformer_function import custom_layer_norm

def make_sinusoidal_emb(s, d, device='cpu', offset=0):
    # s: sequence length, d: dimension of each position's embedding
    # outputs [s, d] positional embedding
    emb = torch.zeros((s, d), device=device)
    denoms = 1e4 ** (torch.arange(0, d, 2).to(device) / d)
    # denoms.shape: (emb_dim / 2)
    timesteps = torch.arange(0, s).to(device) + offset
    # timesteps.shape: (max_seq_len)
    trig_args = timesteps[:, None] / denoms
    # trig_args.shape: (max_seq_len, emb_dim / 2)
    emb[:, ::2] = torch.sin(trig_args)
    emb[:, 1::2] = torch.cos(trig_args)
    return emb * d ** (-0.5)


class PosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, x):
        # x: [batch_size, max_seq_len, emb_dim]
        # returns positional embedding of shape [max_seq_len, emb_dim]
        b, s, d = x.shape
        return make_sinusoidal_emb(s, d, x.device)
        
        
class FFLayer(nn.Module):
    def __init__(self, dim, ff_dim, dropout):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
        )
        
    def forward(self, x):
        return self.ff(x)
        
        
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, dim, dropout, causal=False):
        super().__init__()
        self.W_qkv = nn.Linear(dim, 3*dim, bias=False)
        self.n_heads = n_heads
        if n_heads > 1:
            self.W_o = nn.Linear(dim, dim, bias=False)
        self.scale = (dim / n_heads) ** (-0.5)
        self.dropout = nn.Dropout(dropout)
        self.causal = causal
        
    def forward(self, x, src_mask, x_kv=None):
        # x: (batch_size, max_seq_len, dim)
        # src_mask: (batch_size, max_seq_len)
        if x_kv is None:
            x = self.W_qkv(x)
        else:
            d = self.W_qkv.weight.shape[1]
            q = nn.functional.linear(x, self.W_qkv.weight[:d], self.W_qkv.bias[:d])
            kv = nn.functional.linear(x_kv, self.W_qkv.weight[d:], self.W_qkv.bias[d:])
            x = torch.cat((q, kv), dim=2)
        q, k, v = rearrange(x, 'b s (c n_h d) -> c (b n_h) s d', n_h=self.n_heads, c=3)
        logits = torch.bmm(q, k.transpose(1, 2)) * (q.shape[-1]) ** (-0.5)
        # logits.shape: (batch_size*n_heads, max_seq_len, max_seq_len)
        if src_mask is not None:
            logits = rearrange(logits, '(b n_h) s1 s2 -> b n_h s1 s2', n_h=self.n_heads)
            logits.masked_fill_(src_mask[:, None, None, :], float("-inf"))
            logits = rearrange(logits, 'b n_h s1 s2 -> (b n_h) s1 s2', n_h=self.n_heads)
        self.causal_mask(logits)
        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        out = torch.bmm(attn, v)
        if self.n_heads > 1:
            out = rearrange(out, '(b n_h) s d -> b s (n_h d)', n_h=self.n_heads)
            out = self.W_o(out)
        return out

    def causal_mask(self, logits):
        if not self.causal:
            return
        i, j = logits.shape[-2:]
        causal_mask = torch.ones((i, j), dtype=torch.bool, device=logits.device).triu(j - i + 1)
        logits.masked_fill_(causal_mask, float("-inf"))


class RelativeMultiHeadAttention(MultiHeadAttention):
    def __init__(self, n_heads, dim, dropout, causal=False):
        super().__init__(n_heads, dim, dropout)
        self.global_content_bias = nn.Parameter(torch.zeros((n_heads, 1, dim // n_heads)))
        self.global_pos_bias = nn.Parameter(torch.zeros((n_heads, 1, dim // n_heads)))
        self.pe_to_key = nn.Linear(dim, dim, bias=False)
        self.causal = causal

    def _rel_shift(self, pos):
        # pos: (batch_size, n_heads, msl, 2*msl - 1)
        p = F.pad(pos, (0, 1, 0, 1)).flatten(-2) # (batch_size, n_heads, (msl + 1)*2*msl)
        p = p.narrow(-1, pos.shape[-1] // 2, pos.shape[-1] * pos.shape[-2]).reshape(pos.shape)
        return p.narrow(-1, 0, (pos.shape[-1] + 1) // 2)

    def forward(self, x, src_mask, x_kv=None):
        # x: (batch_size, s + 1, dim) (the + 1 is due to cls token)
        # src_mask: (batch_size, s + 1)
        b, s, d = x.shape; s = s - 1 # subtract 1 due to cls token
        if x_kv is None:
            x = self.W_qkv(x)
        else:
            d = self.W_qkv.weight.shape[1]
            q = nn.functional.linear(x, self.W_qkv.weight[:d], self.W_qkv.bias[:d])
            kv = nn.functional.linear(x_kv, self.W_qkv.weight[d:], self.W_qkv.bias[d:])
            x = torch.cat((q, kv), dim=2)
        q, k, v = rearrange(x, 'b s (c n_h d) -> c b n_h s d', n_h=self.n_heads, c=3) 
        # add q biases
        q_content = q + self.global_content_bias # (batch_size, n_heads, s + 1, d)
        q_pos = q[:, :, 1:] + self.global_pos_bias # (batch_size, n_heads, s, d)

        # compute relative positional embeddings
        pe = make_sinusoidal_emb(2*s - 1, d, x.device, -s + 1)
        pe_key = self.pe_to_key(pe) # (2s - 1, d)
        pe_key = rearrange(pe_key, 'l (n_h d) -> n_h l d', n_h=self.n_heads)

        # compute positional contributions to logits
        # (batch_size, n_heads, s, d) x (n_heads, d, 2s - 1) -> (batch_size, n_heads, s, 2s - 1)
        pos = q_pos @ pe_key.transpose(1, 2)
        pos = self._rel_shift(pos) # (batch_size, n_heads, s, s)

        # compute logits
        logits = q_content @ k.transpose(2, 3) + F.pad(pos, (1, 0, 1, 0))
        logits = logits * (q.shape[-1]) ** (-0.5)
        # logits.shape: (batch_size, n_heads, s + 1, s + 1)

        if src_mask is not None:
            logits.masked_fill_(src_mask[:, None, None, :], float("-inf"))
        self.causal_mask(logits)

        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = rearrange(out, 'b n_h s d -> b s (n_h d)', n_h=self.n_heads)
        if self.n_heads > 1:
            out = self.W_o(out)
        return out
    
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, ff_dim, dropout, pre_ln=False, relative=False,
        custom_ln=False, causal=False, cross=False):
        super().__init__()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        Attention = RelativeMultiHeadAttention if relative else MultiHeadAttention
        self.mha = Attention(n_heads, dim, dropout, causal=causal)

        self.ff_layer = FFLayer(dim, ff_dim, dropout)
        if custom_ln:
            layer_norm = lambda dim, elementwise_affine: custom_layer_norm
        else:
            layer_norm = nn.LayerNorm
        self.norm1 = layer_norm(dim, elementwise_affine=False)
        self.norm2 = layer_norm(dim, elementwise_affine=False)
        if cross:
            self.mha_cross = Attention(n_heads, dim, dropout, causal=False)
            self.norm_cross = layer_norm(dim, elementwise_affine=False)

        self.pre_ln = pre_ln
        self.cross = cross
        
    def forward(self, x, src_mask, x_2=None, src_mask_2=None):
        # x: (batch_size, max_seq_len, dim)
        # src_mask: (batch_size, max_seq_len)
        
        if self.pre_ln:
            x = x + self.dropout1(self.mha(self.norm1(x), src_mask))
            x = x + self.mha_cross(self.norm_cross(x), src_mask_2, self.norm_cross(x_2))
            x = x + self.dropout2(self.ff_layer(self.norm2(x)))
        else:
            x = self.norm1(x + self.dropout1(self.mha(x, src_mask)))
            x = self.norm_cross(x + self.mha_cross(x, src_mask_2, x_2))
            x = self.norm2(x + self.dropout2(self.ff_layer(x)))
        return x


class TransformerModule(nn.Module):
    def __init__(self, dim, n_layers, n_heads, ff_dim, dropout, pre_ln=False, universal=False,
                 relative=False, custom_ln=False, causal=False, cross=False):
        super().__init__()
        args = (dim, n_heads, ff_dim, dropout)
        kwargs = {'pre_ln': pre_ln, 'relative': relative, 'custom_ln': custom_ln, 'causal': causal,
                  'cross': cross}
        if universal:
            self.transformer_block = TransformerBlock(*args, **kwargs)
            self.transformer_blocks = [self.transformer_block for i in range(n_layers)]
        else:
            self.transformer_blocks = nn.ModuleList([TransformerBlock(*args, **kwargs) for i in range(n_layers)])

    def forward(self, x, src_mask, x_2=None, src_mask_2=None):
        for block in self.transformer_blocks:
            x = block(x, src_mask, x_2=x_2, src_mask_2=src_mask_2)
        return x


class Encoder(TransformerModule):
    def __init__(self, dim, n_layers, n_heads, ff_dim, dropout, pre_ln=False, universal=False,
                 relative=False, custom_ln=False, causal=False):
        super().__init__(dim, n_layers, n_heads, ff_dim, dropout, pre_ln=pre_ln, universal=universal,
                         relative=relative, custom_ln=custom_ln, causal=causal, cross=False)


class Decoder(TransformerModule):
    def __init__(self, dim, n_layers, n_heads, ff_dim, dropout, pre_ln=False, universal=False,
                 relative=False, custom_ln=False):
        super().__init__(dim, n_layers, n_heads, ff_dim, dropout, pre_ln=pre_ln, universal=universal,
                         relative=relative, custom_ln=custom_ln, causal=True, cross=True)
        
        
class Transformer(nn.Module):
    def __init__(self, dim, n_layers, n_heads, ff_dim, vocab_size, n_classes, dropout=0.1,
                 pre_ln=False, universal=False, relative=False, emb_norm=False, custom_ln=False,
                 causal=False, padding_idx=None, append_cls=True, autoreg=False, decoder=False,
                 reduce='cls_token'):
        super().__init__()
        if padding_idx is None:
            self.embedding = nn.Embedding(vocab_size+1, dim, padding_idx=vocab_size)
        else:
            self.embedding = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
        self.embedding.weight.data.normal_(0, dim**(-0.5))
        self.append_cls = append_cls
        if append_cls:
            self.cls_token = nn.Parameter(torch.normal(0, dim**(-0.5), (1, 1, dim)))
        self.pos_emb = PosEmb(dim)
        self.emb_dropout = nn.Dropout(dropout)

        self.encoder = Encoder(dim, n_layers, n_heads, ff_dim, dropout, pre_ln=pre_ln, universal=universal,
                               relative=relative, custom_ln=custom_ln, causal=causal)
        if decoder:
            self.decoder = Decoder(dim, n_layers, n_heads, ff_dim, dropout, pre_ln=pre_ln, universal=universal,
                                   relative=relative, custom_ln=custom_ln)

        self.autoreg = (n_classes == 'auto') or autoreg
        if n_classes == 'auto':
            n_classes = vocab_size
        self.linear = nn.Linear(dim, n_classes)
        self.relative = relative
        self.emb_norm = emb_norm
        self.reduce = reduce
        self.inference = False
        
    def forward(self, x, seq_lens=None, src_mask=None, x_2=None, seq_lens_2=None):
        """
        x: (batch_size, max_seq_len)
        seq_lens: (batch_size)
        x_2: (batch_size, max_seq_len_2) (output sequence for encoder-decoder tf)
        seq_lens_2: (batch_size)
        """
        # preparing the input
        x, src_mask = self.prepare_input(x, seq_lens, src_mask, self.append_cls)
        if (x_2 is not None) and (not self.inference):
            # TODO: support padding idx other than vocab_size
            start_pad = torch.full((len(x_2), 1), len(self.embedding.weight) - 1, dtype=torch.long, device=x_2.device)
            x_2 = torch.cat((start_pad, x_2), dim=1)
            x_2, src_mask_2 = self.prepare_input(x_2[:, :-1], seq_lens_2, None, False)

        x = self.encoder(x, src_mask)
        if hasattr(self, 'decoder') and (not self.inference):
            y = self.decoder(x_2, src_mask_2, x_2=x, src_mask_2=src_mask, inference=False)
            logits = self.linear(y)
        if hasattr(self, 'decoder') and self.inference:
            x_out, logits = self.forward_inference(x, src_mask)

        if not hasattr(self, 'decoder'):
            if self.reduce == 'cls_token':
                x = x[:, 0]
            if self.reduce == 'mean':
                x = x.mean(dim=1)
            logits = self.linear(x)

        return logits

    def prepare_input(self, x, seq_lens, src_mask, append_cls):
        x = self.embedding(x)
        x = self.emb_dropout(x)
        if not self.relative:
            x = x = self.pos_emb(x)
        if append_cls:
            x = torch.cat((self.cls_token.repeat(len(x), 1, 1), x), dim=1)
        if self.emb_norm:
            x = custom_layer_norm(x)
        if src_mask is None:
            pos = torch.arange(0, x.shape[1], device=x.device)
            src_mask = seq_lens[:, None] + append_cls <= pos
        return x, src_mask

    def eval(self):
        super(Transformer, self).eval()
        self.inference = True

    def forward_inference(self, x, src_mask, n=None):
        """
        x is the encoded input sequence. Given x, generate an output sequence
        using the decoder. I.e., perform the following steps:
        feed (<pad>, x) to the decoder -> c1
        feed (<pad>c1, x) to the decoder -> c2
        feed (<pad>c1c2, x) to the decoder -> c3
        ...
        until n tokens have been output by the decoder. Then return
        c1c2c3....cn
        """
        if n is None:
            n = x.shape[1]
        # TODO: this actually depends on pad_idx, but currently is assumed to be vocab_size
        x_dec = torch.full((len(x), 1), len(self.embedding.weight) - 1, dtype=torch.long, device=x.device)
        lengths = torch.ones((len(x),), device=x.device)
        x_2, src_mask_2 = self.prepare_input(x_dec, lengths, None, False)
        # TODO: rewrite this loop to avoid redundant computation
        for i in range(n): 
            # TODO: replace None with src_mask for x_dec
            y = self.decoder(x_2, src_mask_2, x, src_mask)
            logits = self.linear(y) # (b, i + 1, n_classes)
            token_probs = logits[:, :].softmax(dim=2)
            x_dec = torch.cat((x_dec, token_probs.argmax(dim=2)[:, None]), dim=1)
            lengths = lengths + 1
            x_2, src_mask_2 = self.prepare_input(x_dec, lengths) 
        return x_dec[:, 1:], logits
