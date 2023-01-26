from einops import rearrange, repeat
import torch
from torch import nn
F = nn.functional

from .implicit_transformer_function import ImplicitTransformerFn, custom_layer_norm

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
    def __init__(self, n_heads, dim, dropout):
        super().__init__()
        self.W_qkv = nn.Linear(dim, 3*dim, bias=False)
        self.n_heads = n_heads
        if n_heads > 1:
            self.W_o = nn.Linear(dim, dim, bias=False)
        self.scale = (dim / n_heads) ** (-0.5)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, src_mask):
        # x: (batch_size, max_seq_len, dim)
        # src_mask: (batch_size, max_seq_len)
        x = self.W_qkv(x)
        x = rearrange(x, 'b s (n_h d) -> (b n_h) s d', n_h=self.n_heads)
        q, k, v = x.chunk(3, dim=2)
        logits = torch.bmm(q, k.transpose(1, 2)) * (q.shape[-1]) ** (-0.5)
        # logits.shape: (batch_size*n_heads, max_seq_len, max_seq_len)
        if src_mask is not None:
            logits = rearrange(logits, '(b n_h) s1 s2 -> b n_h s1 s2', n_h=self.n_heads)
            logits.masked_fill_(src_mask[:, None, None, :], float("-inf"))
            logits = rearrange(logits, 'b n_h s1 s2 -> (b n_h) s1 s2', n_h=self.n_heads)
        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        out = torch.bmm(attn, v)
        if self.n_heads > 1:
            out = rearrange(out, '(b n_h) s d -> b s (n_h d)', n_h=self.n_heads)
            out = self.W_o(out)
        return out


class RelativeMultiHeadAttention(MultiHeadAttention):
    def __init__(self, n_heads, dim, dropout):
        super().__init__(n_heads, dim, dropout)
        self.global_content_bias = nn.Parameter(torch.zeros((n_heads, 1, dim // n_heads)))
        self.global_pos_bias = nn.Parameter(torch.zeros((n_heads, 1, dim // n_heads)))
        self.pe_to_key = nn.Linear(dim, dim, bias=False)

    def _rel_shift(self, pos):
        # pos: (batch_size, n_heads, msl, 2*msl - 1)
        p = F.pad(pos, (0, 1, 0, 1)).flatten(-2) # (batch_size, n_heads, (msl + 1)*2*msl)
        p = p.narrow(-1, pos.shape[-1] // 2, pos.shape[-1] * pos.shape[-2]).reshape(pos.shape)
        return p.narrow(-1, 0, (pos.shape[-1] + 1) // 2)

    def forward(self, x, src_mask):
        # x: (batch_size, s + 1, dim) (the + 1 is due to cls token)
        # src_mask: (batch_size, s + 1)
        b, s, d = x.shape; s = s - 1 # subtract 1 due to cls token
        x = self.W_qkv(x)
        x = rearrange(x, 'b s (n_h d) -> b n_h s d', n_h=self.n_heads)
        q, k, v = x.chunk(3, dim=3) # each is (batch_size, n_heads, s + 1, d)

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

        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = rearrange(out, 'b n_h s d -> b s (n_h d)', n_h=self.n_heads)
        if self.n_heads > 1:
            out = self.W_o(out)
        return out


class ImplicitMultiHeadAttention(nn.Module):
    def __init__(self, n_heads, dim, dropout):
        super().__init__()
        self.W_qkv = nn.Linear(dim, 3*dim, bias=False)
        self.n_heads = n_heads
        if n_heads > 1:
            self.W_o = nn.Linear(dim, dim, bias=False)
        self.scale = (dim / n_heads) ** (-0.5)
        self.dropout = nn.Dropout(dropout)
        self.A = nn.Linear(dim, 3*dim, bias=False)

    def forward(self, x, src_mask):
        return ImplicitTransformerFn.apply(self.A.weight, self.W_qkv.weight, 
            self.W_o.weight, x, src_mask, self.n_heads,
        )
    
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, ff_dim, dropout, pre_ln=False, relative=False,
        implicit=False, custom_ln=False):
        super().__init__()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.implicit = implicit
        if implicit:
            self.mha = ImplicitMultiHeadAttention(n_heads, dim, dropout)
        elif relative:
            self.mha = RelativeMultiHeadAttention(n_heads, dim, dropout)
        else:
            self.mha = MultiHeadAttention(n_heads, dim, dropout)
        self.ff_layer = FFLayer(dim, ff_dim, dropout)
        if custom_ln:
            layer_norm = lambda dim, elementwise_affine: custom_layer_norm
        else:
            layer_norm = nn.LayerNorm
        self.norm1 = layer_norm(dim, elementwise_affine=False)
        self.norm2 = layer_norm(dim, elementwise_affine=False)
        self.pre_ln = pre_ln
        
    def forward(self, x, src_mask):
        # x: (batch_size, max_seq_len, dim)
        # src_mask: (batch_size, max_seq_len)
        
        if self.pre_ln:
            x = x + self.dropout1(self.mha(self.norm1(x), src_mask))
            x = x + self.dropout2(self.ff_layer(self.norm2(x)))
        else:
            if self.implicit:
                x = self.mha(x, src_mask)
            else:
                x = self.norm1(x + self.dropout1(self.mha(x, src_mask)))
            x = self.norm2(x + self.dropout2(self.ff_layer(x)))
        return x
        
        
class Transformer(nn.Module):
    def __init__(self, dim, n_layers, n_heads, ff_dim, vocab_size, n_classes,
                 dropout=0.1, pre_ln=False, universal=False, relative=False,
                 implicit=False, emb_norm=False, custom_ln=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size+1, dim, padding_idx=vocab_size)
        self.embedding.weight.data.normal_(0, dim**(-0.5))
        self.cls_token = nn.Parameter(torch.normal(0, dim**(-0.5), (1, 1, dim)))
        self.pos_emb = PosEmb(dim)
        self.emb_dropout = nn.Dropout(dropout)
        if universal:
            self.transformer_block = TransformerBlock(dim, n_heads, ff_dim, dropout, pre_ln=pre_ln,
                relative=relative)
            self.transformer_blocks = [self.transformer_block for i in range(n_layers)]
        else:
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(dim, n_heads, ff_dim, dropout, pre_ln=pre_ln, relative=relative,
                    implicit=implicit, custom_ln=custom_ln)
                for i in range(n_layers)
            ])
        self.linear = nn.Linear(dim, n_classes)
        self.relative = relative
        self.implicit = implicit
        self.emb_norm = emb_norm
        
    def forward(self, x, seq_lens):
        # x: (batch_size, max_seq_len)
        # seq_lens: (batch_size)
        pos = torch.arange(0, x.shape[1] + 1, device=x.device)
        src_mask = seq_lens[:, None] + 1 <= pos
        x = self.embedding(x)
        x = self.emb_dropout(x)
        if not self.relative:
            x = x + self.pos_emb(x)
        x = torch.cat((self.cls_token.repeat(len(x), 1, 1), x), dim=1)
        if self.implicit or self.emb_norm:
            x = custom_layer_norm(x)
        
        for i in range(len(self.transformer_blocks)):
            x = self.transformer_blocks[i](x, src_mask)
        logits = self.linear(x[:, 0, :])
        return logits