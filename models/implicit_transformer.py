import numpy as np
from sklearn.decomposition import TruncatedSVD
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat
import time

from models.implicit_transformer_function import custom_layer_norm, ITFn2
from models.transformer import PosEmb
from models.projection import mrs_project, project_l2, l1_project, solve_NA_qkv_opt, create_NA, create_A_qkv_norms, L_qkv


def scale_by_S(A, A_qkv, S, n_heads, ln_dim, ln_offset):
    A = A.data.clone().detach().cpu()
    A_qkv = A_qkv.clone().detach().cpu()
    if S is None:
        return A, A_qkv
    A[:, :len(S)] = (1/S[:A.shape[0], None]) * A[:, :len(S)] * S
    A_qkv = rearrange(A_qkv, '(n_h n_m d_2) d_1 -> n_m (n_h d_2) d_1', n_h=n_heads, n_m=3)
    A_qkv[2] *= (1/S[A.shape[0]:, None])
    A_qkv = rearrange(A_qkv, 'n_m (n_h d_2) d_1 -> (n_h n_m d_2) d_1', n_h=n_heads)
    A_qkv = A_qkv * S[ln_offset:ln_offset + ln_dim]
    return A, A_qkv


class ImplicitTransformer(nn.Module):
    def __init__(self, dim, n_heads, vocab_size, n_classes, relu_dim=None, ln_dim=None,
        emb_dim=None, n_layer_norms=1, n_relu_heads=None):
        super().__init__()
        relu_dim = dim if relu_dim is None else relu_dim
        ln_dim = dim if ln_dim is None else ln_dim
        emb_dim = dim if emb_dim is None else emb_dim

        self.embedding = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=vocab_size)
        self.pos_emb = PosEmb(emb_dim)
        self.A = nn.Parameter(torch.empty((relu_dim + ln_dim, relu_dim + ln_dim + dim + emb_dim)))
        self.b = nn.Parameter(torch.zeros((relu_dim + ln_dim,)))
        self.A_qkv = nn.Parameter(torch.empty((3*dim, ln_dim)))
        self.n_heads = n_heads
        self.n_layer_norms = n_layer_norms
        self.n_relu_heads = n_relu_heads
        self.cls_token = nn.Parameter(torch.normal(0, emb_dim**(-0.5), (1, 1, emb_dim)))
        self.cls_linear = nn.Linear(relu_dim + ln_dim + dim, n_classes, bias=True)

        self.relu_dim = relu_dim
        self.ln_dim = ln_dim
        self.dim = dim
        self.emb_dim = emb_dim
        self.S = None

        torch.nn.init.kaiming_normal_(self.A)
        torch.nn.init.kaiming_normal_(self.A_qkv)
        #self.project(v=0.95, exact=True)

    def forward(self, x, seq_lens):
        # setup inputs to implicit model
        pos = torch.arange(0, x.shape[1] + 1, device=x.device)
        src_mask = seq_lens[:, None] + 1 <= pos
        u = self.embedding(x)
        u = u + self.pos_emb(u)
        u = torch.cat((self.cls_token.repeat(len(u), 1, 1), u), dim=1)

        # implicit time!!
        x = ITFn2.apply(self.A, self.A_qkv, self.b, u, src_mask, self.n_heads, self.n_layer_norms,
            self.relu_dim, self.ln_dim, self.dim, self.emb_dim)
        logits = self.cls_linear(x[:, 0])
        return logits

    def get(self, i, j):
        i_chunks = np.cumsum([0] + [self.relu_dim, self.ln_dim])
        j_chunks = np.cumsum([0] + [self.relu_dim, self.ln_dim, self.dim, self.emb_dim])
        return self.A[i_chunks[i]:i_chunks[i+1], j_chunks[j]:j_chunks[j+1]]

    def project(self, v=0.95, exact=True):
        relu_dim, ln_dim, attn_dim = self.relu_dim, self.ln_dim, self.dim
        n_layer_norms, n_heads = self.n_layer_norms, self.n_heads
        A, A_qkv = scale_by_S(self.A, self.A_qkv, self.S, self.n_heads, self.ln_dim, self.relu_dim)
        p = A_qkv.shape[0] // (3*n_heads)

        time1 = time.time()
        # compute norm matrices
        NA = create_NA(A, relu_dim, ln_dim, attn_dim, n_layer_norms, n_heads)
        old_NA = NA.clone()
        A_qkv_norms_0 = create_A_qkv_norms(A_qkv, n_heads, n_layer_norms)

        time2 = time.time()
        print(f"Time to compute norm matrices: {time2 - time1}")
        # project norm matrices
        mrs_project(NA, 0.95)
        sol, A_qkv_norms = solve_NA_qkv_opt(A_qkv_norms_0, p, 0.95)

        time3 = time.time()
        print(f"Time to project norm matrices: {time3 - time2}")

        # project parameter matrices using projected norms
        A[:relu_dim, :relu_dim] = NA[:relu_dim, :relu_dim] * torch.sign(A[:relu_dim, :relu_dim])

        tol = 1e-6
        for i in range(n_layer_norms):
            dim = ln_dim // n_layer_norms
            # ln to relu
            A[:relu_dim, relu_dim + i*dim:relu_dim + (i+1)*dim] *= \
                (NA[:relu_dim, relu_dim + i] / torch.clip(old_NA[:relu_dim, relu_dim + i], 1e-8))[:, None]
            
            # relu to ln
            A[relu_dim + i*dim:relu_dim + (i+1)*dim, :relu_dim] *= \
                NA[relu_dim + i, :relu_dim] / torch.clip(old_NA[relu_dim + i, :relu_dim], 1e-8)

            for j in range(n_layer_norms):
                # ln to ln
                if NA[relu_dim + i, relu_dim + j] + tol < old_NA[relu_dim + i, relu_dim + j]:
                    project_l2(
                        A[relu_dim + i*dim:relu_dim + (i+1)*dim, relu_dim + j*dim:relu_dim + (j+1)*dim],
                        NA[relu_dim + i, relu_dim + j].item(), 30, exact=exact,
                    )
                
                
            assert p == attn_dim // n_heads
            offset = relu_dim + ln_dim
            for j in range(n_heads):
                # attn to ln
                if (NA[relu_dim + i, relu_dim + n_layer_norms + j] + tol < 
                    old_NA[relu_dim + i, relu_dim + n_layer_norms + j]):
                    project_l2(
                        A[relu_dim + i*dim:relu_dim + (i+1)*dim, offset + j*p:offset + (j+1)*p],
                        NA[relu_dim + i, relu_dim + n_layer_norms + j].item(), 30, exact=exact,
                    )

        for i in range(n_heads):
            # attn to relu
            A[:relu_dim, offset + i*p:offset + (i+1)*p] *= \
                (NA[:relu_dim, relu_dim + n_layer_norms + i] / 
                 torch.clip(old_NA[:relu_dim, relu_dim + n_layer_norms + i], 1e-8)
                )[:, None]
            
        A_qkv = rearrange(A_qkv, '(n_h n_m d2) (n_l d1) -> n_m n_h n_l d2 d1', 
                          n_h=n_heads, n_m=3, n_l=n_layer_norms)

        for i in range(n_heads):
            for j in range(n_layer_norms):
                for k in range(3):
                    if A_qkv_norms_0[k, i, j] > tol + A_qkv_norms[k, i, j]:
                        project_l2(A_qkv[k, i, j], A_qkv_norms[k, i, j], 30, exact=exact)
        A_qkv = rearrange(A_qkv, 'n_m n_h n_l d2 d1 -> (n_h n_m d2) (n_l d1)')
        inv_S = 1 / self.S if torch.is_tensor(self.S) else None
        A, A_qkv = scale_by_S(A, A_qkv, inv_S, self.n_heads, self.ln_dim, self.relu_dim)
        self.A.data = A.to(self.A.data.device)
        self.A_qkv.data = A_qkv.to(self.A_qkv.data.device)

    def port_transformer(self, dim, ff_dim, n_layers, transformer):
        A = self.A.data
        A_qkv = self.A_qkv.data
        b = self.b.data
        A.zero_()
        A_qkv.zero_()
        b.zero_()
        relu_n = n_layers*ff_dim
        ln_n = (1+2*n_layers)*dim
        attn_n = n_layers*dim
        
        ln_offset = self.relu_dim
        attn_offset = self.relu_dim + self.ln_dim
        U_offset = self.relu_dim + self.ln_dim + self.dim
        A[ln_offset:ln_offset + dim, U_offset:U_offset + dim] = torch.eye(dim)
        for i, block in enumerate(transformer.transformer_blocks):
            W_qkv = block.mha.W_qkv.weight.data.clone()
            W_o = block.mha.W_o.weight.data.clone()
            W1 = block.ff_layer.ff[0].weight.data.clone()
            W2 = block.ff_layer.ff[3].weight.data.clone()
            b1 = block.ff_layer.ff[0].bias.data.clone()
            b2 = block.ff_layer.ff[3].bias.data.clone()
            # 2*ith ln feeds to ith attn
            # having multiple lns messes with Lipschitz constant?
            A_qkv[i*3*dim:(i+1)*3*dim, 2*i*dim:(2*i+1)*dim] = W_qkv 

            # ith attn feeds to 2*i+1th ln
            A[ln_offset + (2*i+1)*dim:ln_offset + (2*i+2)*dim,
              attn_offset + i*dim:attn_offset + (i+1)*dim] = W_o
            # skip connection from 2*ith ln
            A[ln_offset + (2*i+1)*dim: ln_offset + (2*i+2)*dim,
              ln_offset + 2*i*dim:ln_offset + (2*i+1)*dim] = torch.eye(dim)

            # 2*i+1th ln feeds to ith relu
            A[i*ff_dim:(i+1)*ff_dim, ln_offset + (2*i+1)*dim:ln_offset + (2*i+2)*dim] = W1

            # ith relu feeds to 2*i+2th ln
            A[ln_offset + (2*i+2)*dim: ln_offset + (2*i+3)*dim, i*ff_dim:(i+1)*ff_dim] = W2
            # skip connection from 2*i+1th ln
            A[ln_offset + (2*i+2)*dim:ln_offset + (2*i+3)*dim, 
              ln_offset + (2*i+1)*dim:ln_offset + (2*i+2)*dim] = torch.eye(dim)

            # add bias1 to ith relu
            b[i*ff_dim:(i+1)*ff_dim] = b1
            # add bias2 to 2*i+2th ln
            b[ln_offset + (2*i+2)*dim:ln_offset + (2*i+3)*dim] = b2
        self.embedding.weight.data = transformer.embedding.weight.data.clone()
        self.cls_token.data = transformer.cls_token.data.clone()
        self.cls_linear.weight.data.zero_()
        self.cls_linear.bias.data.zero_()
        self.cls_linear.weight.data[:, attn_offset - dim:attn_offset] = transformer.linear.weight.data.clone()
        self.cls_linear.bias.data = transformer.linear.bias.data.clone()

    def S_from_ported_transformer(self, n_layers, v):
        def assign(S, cs, offset, val):
            S[offset:offset + cs] = val

        S = torch.zeros(self.relu_dim + self.ln_dim + self.dim)
        A = self.A.data.detach().clone().cpu()
        A_qkv = self.A_qkv.data.detach().clone().cpu()
        d = 256
        hd = d // 8
        n_heads = self.n_heads
        n_lns = self.n_layer_norms
        ff_dim = self.relu_dim // n_layers

        ln_offset = self.relu_dim
        attn_offset = self.relu_dim + self.ln_dim
        cache = {}

        assign(S, d, self.relu_dim, 1)
        s_lst = [max([L_qkv(A_qkv, i, 0, n_heads, n_lns, cache) for i in range(8)])]
        s = s_lst[-1] / v
        # why did I do this again (take the max L_qkv over the heads in attn layer)?
        # oh because all these heads together multiplied by W_o route to the next ln?
        for i in range(n_layers):
            assign(S, d, attn_offset + i*d, s)
            s_lst.append(sum([torch.linalg.norm(
                A[ln_offset + (2*i+1)*d:ln_offset + (2*i+2)*d,
                attn_offset + i*d + j*hd:attn_offset + i*d + (j+1)*hd], ord=2) 
                for j in range(8)]))
            s = s*s_lst[-1] / v

            assign(S, d, ln_offset + (2*i + 1)*d, s)
            s_lst.append(A[
                i*ff_dim:(i+1)*ff_dim,
                ln_offset + (2*i+1)*d:ln_offset + (2*i+2)*d
                ].norm(dim=1).max())
            s = s*s_lst[-1] / v
                
            assign(S, ff_dim, i*ff_dim, s)
            s_lst.append(A[ln_offset + (2*i+2)*d:ln_offset + (2*i+3)*d,
                  i*ff_dim:(i+1)*ff_dim].norm(dim=0).sum())
            s = s*s_lst[-1] / v
            
            assign(S, d, ln_offset + (2*i + 2)*d, s)
            if i < n_layers - 1:
                s_lst.append(max([L_qkv(A_qkv, j + 8*(i+1), 2*i + 2, n_heads, n_lns, cache) for j in range(8)]))
                s = s**3 * s_lst[-1] / v
        self.S = S