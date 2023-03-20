from einops import rearrange
import numpy as np
import scipy
from sklearn.decomposition import TruncatedSVD
import torch

def project_l2(W, s_max, k, exact=False, numpy=False):
    # W: a 2D torch.tensor
    if s_max <= 1e-6:
        W.data.zero_()
        return
    W_cpu = W.data.cpu()
    if numpy:
        W_numpy = W.data.cpu().numpy()
        if exact:
            u, s, v = np.linalg.svd(W_numpy, full_matrices=False)
        else:
            tsvd = TruncatedSVD(k).fit(W_numpy)
            v = tsvd.components_
            s = tsvd.singular_values_
    else:
        u, s, v = torch.svd(W_cpu)
        v = v.transpose(0, 1)
    greater_than_s_max = s > s_max
    if greater_than_s_max.sum() == 0:
        return W
    v_greater = v[greater_than_s_max]
    s_greater = s[greater_than_s_max]
    if numpy:
        s_u_greater = W_numpy @ v_greater.T
    else:
        s_u_greater = W_cpu @ v_greater.T
    s_max_minus_s_u_greater = s_u_greater * (s_max / s_greater - 1)
    if numpy:
        W.data.copy_(torch.tensor(W_numpy + s_max_minus_s_u_greater @ v_greater, device=W.device))
    else:
        W.data.copy_((W_cpu + s_max_minus_s_u_greater @ v_greater).to(W.device))

def l1_project(a_orig, v):
    a_sign = np.sign(a_orig)
    a_abs = np.abs(a_orig)
    a = np.sort(a_abs)

    s = np.sum(a) - v
    l = float(len(a))
    for i in range(len(a)):
        # proposal: alpha <= a[i]
        if s / l > a[i]:
            s -= a[i]
            l -= 1
        else:
            break
    alpha = s / l
    a = a_sign * np.maximum(a_abs - alpha, 0)
    # verify
    assert np.abs(np.abs(a).sum() - v) < 1e-3, (np.abs(a).sum(), v)
    return a

def mrs_project(A, v=0.95):
    # Fangda's code
    A_np = A.clone().detach().cpu().numpy()
    x = np.abs(A_np).sum(axis=-1)
    where = np.where(x > v)[0]
    for idx in where:
        # read the vector
        a_orig = A_np[idx, :]
        a = l1_project(a_orig, v)
        # write back
        A_np[idx, :] = a
    # 0.0
    #A_np = np.zeros_like(A_np)
    A.data.copy_(torch.tensor(A_np, dtype=A.dtype, device=A.device))
    return where

def solve_NA_qkv_opt(A_qkv_norms_0, p, v=0.95):
    tol = 1e-6
    n_heads = A_qkv_norms_0.shape[1]
    n_layer_norms = A_qkv_norms_0.shape[2]
    A_qkv_sums_0 = A_qkv_norms_0.sum(axis=2)
    Q_s, K_s, V_s = A_qkv_sums_0
    constrs = V_s*(2*Q_s*K_s/p**0.5 + 1)
    where = np.where(constrs > v + tol)[0]
    if len(where) == 0:
        return None, A_qkv_norms_0
    A_qkv_sums_0 = A_qkv_sums_0[:, where]
    def obj_fn(A_qkv_sums):
        A_qkv_sums = A_qkv_sums.reshape(3, -1)
        return np.sum((A_qkv_sums - A_qkv_sums_0)**2)

    def obj_jac(A_qkv_sums):
        return 2*(A_qkv_sums - A_qkv_sums_0.flatten())

    def obj_hess(A_qkv_sums):
        return 2*np.eye(len(A_qkv_sums))

    def constraint(A_qkv_sums):
        Q_s, K_s, V_s = np.array_split(A_qkv_sums, 3)
        return V_s*(2*Q_s*K_s/p**0.5 + 1)

    def constraint_jac(A_qkv_sums):
        Q_s, K_s, V_s = np.array_split(A_qkv_sums, 3)
        H = len(Q_s)
        buffer = np.zeros((H, 3, H))
        jac = np.vstack(
            (2*V_s*K_s/p**0.5,
             2*V_s*Q_s/p**0.5,
             2*Q_s*K_s/p**0.5 + 1)
        )
        jac = rearrange(jac, 'm H -> H m 1')
        index = np.arange(H).reshape(H, 1, 1)
        np.put_along_axis(buffer, index, jac, axis=2)
        return buffer.reshape(H, 3*H)

    constraints = [
        scipy.optimize.NonlinearConstraint(constraint, -np.inf, v,
                                          jac=constraint_jac)
    ]

    bounds = scipy.optimize.Bounds(0, A_qkv_sums_0.flatten())
    sol = scipy.optimize.minimize(obj_fn, A_qkv_sums_0.flatten(), method='trust-constr',
                              bounds=bounds, constraints=constraints, jac=obj_jac,
                              hess=obj_hess)
    A_qkv_sums_flat = sol.x
    A_qkv_sums_0_flat = A_qkv_sums_0.flatten()
    index = np.where(A_qkv_sums_0_flat > A_qkv_sums_flat)[0]
    A_qkv_norms = np.copy(A_qkv_norms_0)
    A_qkv_norms_sel = A_qkv_norms[:, where].reshape(3*len(where), n_layer_norms)
    for qkv_sum, idx in zip(A_qkv_sums_flat, index):
        a_orig = A_qkv_norms_sel[idx, :]
        a = l1_project(a_orig, qkv_sum)
        A_qkv_norms_sel[idx, :] = a
    A_qkv_norms_sel = A_qkv_norms_sel.reshape(3, len(where), n_layer_norms)
    A_qkv_norms[:, where] = A_qkv_norms_sel
    return sol, A_qkv_norms

def get_block(A, i, i_dim, j, j_dim, i_offset, j_offset):
    i_base = i_offset + i*i_dim
    j_base = j_offset + j*j_dim
    return A[i_base:i_base + i_dim, j_base:j_base + j_dim]

def create_NA(A, relu_dim, ln_dim, attn_dim, n_layer_norms, n_heads, n_relu_heads):
    A = A.data.detach().clone().cpu()
    A_norm = torch.zeros(
        (n_relu_heads + n_layer_norms, n_relu_heads + n_layer_norms + n_heads)
    )
    chunks = np.cumsum([relu_dim // n_relu_heads]*n_relu_heads
        + [ln_dim // n_layer_norms]*n_layer_norms
        + [attn_dim // n_heads]*n_heads)
    chunks = tuple(chunks)
    A = A.tensor_split(chunks[:n_relu_heads + n_layer_norms - 1], dim=0)
    A = [A_s.tensor_split(chunks, dim=1)[:-1] for A_s in A]
    assert np.all([len(A_s) == n_relu_heads + n_layer_norms + n_heads for A_s in A])
    assert len(A) == n_relu_heads + n_layer_norms
    for i in range(n_relu_heads + n_layer_norms):
        for j in range(n_relu_heads + n_layer_norms + n_heads):
            A_norm[i, j] = torch.linalg.norm(A[i][j], ord=2)
    return A_norm

def create_A_qkv_norms(A_qkv, n_heads, n_layer_norms):
    A_qkv = A_qkv.data.clone().detach().cpu()
    A_qkv = rearrange(A_qkv, '(n_h n_m d1) (n_ln d2) -> n_h n_ln n_m d1 d2',
                     n_h=n_heads, n_ln=n_layer_norms, n_m=3)
    A_qkv_norms = torch.zeros((3, n_heads, n_layer_norms))
    for i in range(n_heads):
        for j in range(n_layer_norms):
            for k in range(3):
                A_qkv_norms[k, i, j] = torch.linalg.norm(A_qkv[i, j, k], ord=2)
    return A_qkv_norms.numpy()

def L_qkv(A_qkv, i, j, n_heads, n_lns, cache):
    # i: index of head
    # j: index of layer norm input
    A_qkv = rearrange(A_qkv, '(n_h n_m d2) d1 -> n_h n_m d2 d1', n_h=n_heads, n_m=3)
    A_q, A_k, A_v = A_qkv[i]
    d = A_q.shape[-1] // n_lns
    p = A_q.shape[0]
    N_q = torch.linalg.norm(A_q[:, j*d:(j+1)*d], ord=2)
    N_k = torch.linalg.norm(A_k[:, j*d:(j+1)*d], ord=2)
    N_v = torch.linalg.norm(A_v[:, j*d:(j+1)*d], ord=2)
    if i in cache:
        N_q_s, N_k_s, N_v_s = cache[i]
    else:
        N_q_s = sum([torch.linalg.norm(A_q[:, k*d:(k+1)*d], ord=2) for k in range(n_lns)])
        N_k_s = sum([torch.linalg.norm(A_k[:, k*d:(k+1)*d], ord=2) for k in range(n_lns)])
        N_v_s = sum([torch.linalg.norm(A_v[:, k*d:(k+1)*d], ord=2) for k in range(n_lns)])
        cache[i] = (N_q_s, N_k_s, N_v_s)
    return N_v_s*(N_k_s*N_q + N_q_s*N_k)*p**(-0.5) + N_v

# ====================================================================================================
# DEPRECATED
# ====================================================================================================
def create_NA_qkv(A_qkv, n_heads, n_lns):
    NA_qkv = torch.zeros((n_heads, n_lns))
    cache = {}
    for i in range(n_heads):
        for j in range(n_lns):
            NA_qkv[i, j] = L_qkv(A_qkv, i, j, n_heads, n_lns, cache).item()
    return NA_qkv
# ====================================================================================================
# ====================================================================================================