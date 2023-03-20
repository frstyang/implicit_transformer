import torch
F = torch.nn.functional
from functools import partial
import math

from einops import rearrange


def custom_layer_norm(x, n_layer_norms=1):
    chunks = list(x.chunk(n_layer_norms, dim=-1))
    for i, x in enumerate(chunks):
        x = x - x.mean(dim=-1, keepdim=True)
        x = x / ((x**2).sum(dim=-1, keepdim=True) + 1)**(0.5)
        chunks[i] = x
    x = torch.cat(chunks, dim=-1)
    return x


def picard(fn, A, B, U, max_iter=100, tol=1e-3, output_size=None, fn_kwargs={}):
    if output_size is None:
        output_size = U.shape

    fn = partial(fn, **fn_kwargs)

    X = torch.zeros(output_size, device=U.device, dtype=U.dtype)
    for i in range(max_iter):
        X_prev = X
        X = fn(A, B, X, U)
        err = (X - X_prev).norm()
        if err < tol:
            break
    if err >= tol:
        print(f"Failed to converge with ||X - X_prev||_fro = {err:.5f}")
    return X


def transformer_fn(A, B, X, U, src_mask=None, n_heads=8):
    # A: (dim, 3*dim), contains [[A^Q_h A^K_h A^V_h]_{h\in n_heads}]
    # B: (W^{QKV}, W^O)
    #     W^{QKV}: (dim, 3*dim), contains [[W^Q_h W^K_h W^V_h]_{h\in n_heads}]
    #     W^O: (dim, dim), projects concatenated attention heads
    # X: (batch_size, seq_len, dim)
    # U: (batch_size, seq_len, dim)

    W_qkv, W_o = B
    BU = F.linear(U, W_qkv) # (batch_size, seq_len, 3*dim)
    BU = rearrange(BU, 'b s (n_h d) -> (b n_h) s d', n_h=n_heads)
    BU_Q, BU_K, BU_V = BU.chunk(3, dim=2)

    AX = F.linear(X, A)
    AX = rearrange(AX, 'b s (n_h d) -> (b n_h) s d', n_h=n_heads)
    AX_Q, AX_K, AX_V = AX.chunk(3, dim=2)

    q, k, v = BU_Q + AX_Q, BU_K + AX_K, BU_V + AX_V

    logits = torch.bmm(q, k.transpose(1, 2)) / (U.shape[-1] // n_heads) ** (0.5)
    # logits.shape: (batch_size*n_heads, max_seq_len, max_seq_len)
    if src_mask is not None:
        logits = rearrange(logits, '(b n_h) s1 s2 -> b n_h s1 s2', n_h=n_heads)
        logits.masked_fill(src_mask[:, None, None, :], float("-inf"))
        logits = rearrange(logits, 'b n_h s1 s2 -> (b n_h) s1 s2', n_h=n_heads)

    attn = torch.softmax(logits, dim=-1)
    #attn = self.dropout(attn)
    out = torch.bmm(attn, v)
    if n_heads > 1:
        out = rearrange(out, '(b n_h) s d -> b s (n_h d)', n_h=n_heads)
        out = F.linear(out, W_o)
    return custom_layer_norm(U + out)


class ImplicitTransformerFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A, W_qkv, W_o, U, src_mask=None, n_heads=8):
        """
        U: (batch_size, seq_len, dim)
        A: (dim, 3*dim)
        W_qkv: (dim, 3*dim)
        W_o: (dim, dim)
        """
        B = (W_qkv, W_o)
        kwargs = {'src_mask': src_mask, 'n_heads': n_heads}
        with torch.no_grad():
            X = picard(transformer_fn, A, B, U, fn_kwargs=kwargs)
        ctx.save_for_backward(A, W_qkv, W_o, X, U)
        ctx.kwargs = kwargs
        return X

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: (batch_size, seq_len, dim) = dL/dX
        """
        A, W_qkv, W_o, X, U = ctx.saved_tensors
        grad_transformed = torch.zeros_like(X)
        # conduct a picard iteration so grad_transformed equals dL/dX (I - \partial f/\partial X)^{-1}
        A = A.detach().clone()
        W_qkv = W_qkv.detach().clone()
        W_o = W_o.detach().clone()
        X = X.detach().clone().requires_grad_()
        U = U.detach().clone()
        with torch.enable_grad():
            out = transformer_fn(A, (W_qkv, W_o), X, U, **ctx.kwargs)

        def iterate(dummy1, dummy2, g_t, g):
            # dummy1, dummy2 will be set to None
            # g will be set to grad_output (dL/dX)
            # g_t will be dL/dX (I - \partial f/\partial X)^{-1}
            out.backward(g_t, retain_graph=True)
            g_t_df_dx = X.grad.clone()
            X.grad.zero_()
            return g_t_df_dx + g

        grad_transformed = picard(iterate, None, None, grad_output)
        out.backward(torch.zeros_like(grad_transformed), retain_graph=False)

        A = A.requires_grad_()
        W_qkv = W_qkv.requires_grad_()
        W_o = W_o.requires_grad_()
        X = X.detach()
        U = U.requires_grad_()
        with torch.enable_grad():
            out = transformer_fn(A, (W_qkv, W_o), X, U, **ctx.kwargs)
        out.backward(grad_transformed)

        return A.grad, W_qkv.grad, W_o.grad, U.grad, None, None, None, None # fill with Nones in case **kwargs is nonempty


def transformer_fn2(A, B, X, U, src_mask, n_heads, n_layer_norms, relu_dim, ln_dim, dim, emb_dim):
    W_qkv, b = B
    _, ln_x, _ = X.tensor_split([relu_dim, relu_dim + ln_dim], dim=-1)
    cat_x = torch.cat((X, U), dim=-1)

    preact_x = F.linear(cat_x, A, b)
    preact_x1, preact_x2 = preact_x.tensor_split([relu_dim], dim=-1)
    new_relu_x = F.relu(preact_x1)
    new_ln_x = custom_layer_norm(preact_x2, n_layer_norms=n_layer_norms)

    qkv = F.linear(ln_x, W_qkv)
    qkv = rearrange(qkv, 'b s (h d) -> b h s d', h=n_heads)
    q, k, v = qkv.chunk(3, dim=-1)

    logits = torch.matmul(q, k.transpose(-1, -2)) / (q.shape[-1]) ** (0.5)
    # logits.shape: (batch_size, n_heads, max_seq_len, max_seq_len)
    if src_mask is not None:
        logits.masked_fill_(src_mask[:, None, None, :], float("-inf"))

    attn = torch.softmax(logits, dim=-1)
    new_attn_x = torch.matmul(attn, v)
    new_attn_x = rearrange(new_attn_x, 'b n_h s d -> b s (n_h d)')
    return torch.cat((new_relu_x, new_ln_x, new_attn_x), dim=-1)

class ITFn2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A, W_qkv, b, U, src_mask, n_heads, n_layer_norms, relu_dim, ln_dim, dim, emb_dim, jac_reg):
        output_size = (U.shape[0], U.shape[1], relu_dim + ln_dim + dim)
        kwargs = {'src_mask': src_mask, 'n_heads': n_heads, 'n_layer_norms': n_layer_norms,
        'relu_dim': relu_dim, 'ln_dim': ln_dim, 'dim': dim, 'emb_dim': emb_dim}
        with torch.no_grad():
            X = picard(transformer_fn2, A, (W_qkv, b), U, fn_kwargs=kwargs, output_size=output_size)
        ctx.save_for_backward(A, W_qkv, b, X, U)
        ctx.kwargs = kwargs
        ctx.jac_reg = jac_reg
        return X

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: (batch_size, seq_len, dim) = dL/dX
        """
        A, W_qkv, b, X, U = ctx.saved_tensors
        grad_transformed = torch.zeros_like(X)
        # conduct a picard iteration so grad_transformed equals g_t = dL/dX (I - \partial f/\partial X)^{-1}
        # using the fact that g_t = g_t \partial f/\partial x + dL/dX
        A = A.detach().clone()
        W_qkv = W_qkv.detach().clone()
        b = b.detach().clone()
        X = X.detach().clone().requires_grad_()
        U = U.detach().clone()
        with torch.enable_grad():
            out = transformer_fn2(A, (W_qkv, b), X, U, **ctx.kwargs)

        def iterate(dummy1, dummy2, g_t, g):
            # dummy1, dummy2 will be set to None
            # g will be set to grad_output (dL/dX)
            # g_t will be dL/dX (I - \partial f/\partial X)^{-1}
            out.backward(g_t, retain_graph=True)
            g_t_df_dx = X.grad.clone()
            X.grad.zero_()
            return g_t_df_dx + g

        grad_transformed = picard(iterate, None, None, grad_output)
        out.backward(torch.zeros_like(grad_transformed), retain_graph=False)

        A = A.requires_grad_()
        W_qkv = W_qkv.requires_grad_()
        b = b.requires_grad_()
        if not ctx.jac_reg:
            X = X.detach()
        X = X.requires_grad_()
        U = U.requires_grad_()
        with torch.enable_grad():
            out = transformer_fn2(A, (W_qkv, b), X, U, **ctx.kwargs)
        if not ctx.jac_reg:
            out.backward(grad_transformed)

        if ctx.jac_reg:
            def normalize(x):
                return x / x.norm(dim=2, keepdim=True)
            w = normalize(torch.normal(0, 1, X.shape, device=X.device))
            wJ = torch.autograd.grad(out, X, w, create_graph=True)[0]
            def norm(x):
                return (x**2).sum(dim=2)**(0.5)
            with torch.enable_grad():
                nwJ = norm(wJ)
                lse = torch.logsumexp(nwJ, dim=1)
                j_reg = 10*lse.mean()

            torch.autograd.backward((out, j_reg), (grad_transformed, None))

        return A.grad, W_qkv.grad, b.grad, U.grad, None, None, None, None,\
            None, None, None, None, None, None, None # fill with Nones in case **kwargs is nonempty