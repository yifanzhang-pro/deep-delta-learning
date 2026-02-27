import triton.language as tl
import triton
import torch
import torch.nn.functional as F
import math


@triton.jit
def _forward_kernel_TritonEinsum(a_ptr, b_ptr, c_ptr, N, D, s_an, s_ad, s_av, s_bd, s_bk, s_bv, s_cn, s_cd, s_ck, BN: tl.constexpr, BD: tl.constexpr):
    rn, rd = tl.program_id(0) * BN + tl.arange(0, BN), tl.program_id(1) * BD + tl.arange(0, BD)
    rk, rv = tl.arange(0, 4), tl.arange(0, 4)
    a = tl.load(a_ptr + (rn[:, None, None]*s_an + rd[None, :, None]*s_ad + rv[None, None, :]*s_av), 
                mask=(rn[:, None, None] < N) & (rd[None, :, None] < D), other=0.0)
    b = tl.load(b_ptr + (rd[None, :, None, None]*s_bd + rk[None, None, :, None]*s_bk + rv[None, None, None, :]*s_bv), 
                mask=(rd[None, :, None, None] < D), other=0.0)
    c = tl.sum(a[:, :, None, :] * b, axis=3)
    tl.store(c_ptr + (rn[:, None, None]*s_cn + rd[None, :, None]*s_cd + rk[None, None, :]*s_ck), 
             c, mask=(rn[:, None, None] < N) & (rd[None, :, None] < D))

def get_grad_a_configs_TritonEinsum():
    return [
        triton.Config({'BN': 32, 'BD': 32}, num_warps=4),
        triton.Config({'BN': 64, 'BD': 32}, num_warps=4),
        triton.Config({'BN': 128, 'BD': 32}, num_warps=8),
        triton.Config({'BN': 128, 'BD': 64}, num_warps=8),
    ]

@triton.autotune(
    configs=get_grad_a_configs_TritonEinsum(),
    key=['N', 'D'],
)
@triton.jit
def _grad_a_kernel_TritonEinsum(gc_ptr, b_ptr, ga_ptr, N, D, s_cn, s_cd, s_ck, s_bd, s_bk, s_bv, s_gan, s_gad, s_gav, BN: tl.constexpr, BD: tl.constexpr):
    rn, rd = tl.program_id(0) * BN + tl.arange(0, BN), tl.program_id(1) * BD + tl.arange(0, BD)
    rk, rv = tl.arange(0, 4), tl.arange(0, 4)
    gc = tl.load(gc_ptr + (rn[:, None, None]*s_cn + rd[None, :, None]*s_cd + rk[None, None, :]*s_ck), 
                 mask=(rn[:, None, None] < N) & (rd[None, :, None] < D), other=0.0)
    b = tl.load(b_ptr + (rd[None, :, None, None]*s_bd + rk[None, None, :, None]*s_bk + rv[None, None, None, :]*s_bv), 
                mask=(rd[None, :, None, None] < D), other=0.0)
    ga = tl.sum(gc[:, :, :, None] * b, axis=2)
    tl.store(ga_ptr + (rn[:, None, None]*s_gan + rd[None, :, None]*s_gad + rv[None, None, :]*s_gav), 
             ga, mask=(rn[:, None, None] < N) & (rd[None, :, None] < D))


def get_grad_b_configs_TritonEinsum():
    return [
        triton.Config({'BN': 32, 'BD': 32, 'BLOCK_N_SIZE': 256}, num_warps=4),
        triton.Config({'BN': 64, 'BD': 32, 'BLOCK_N_SIZE': 256}, num_warps=4),
        triton.Config({'BN': 128, 'BD': 32, 'BLOCK_N_SIZE': 512}, num_warps=8),
        triton.Config({'BN': 128, 'BD': 32, 'BLOCK_N_SIZE': 512}, num_warps=8),
        triton.Config({'BN': 32, 'BD': 64, 'BLOCK_N_SIZE': 256}, num_warps=4),
    ]

@triton.autotune(
    configs=get_grad_b_configs_TritonEinsum(),
    key=['N', 'D'],
)
@triton.jit
def _grad_b_kernel_TritonEinsum(gc_ptr, a_ptr, gb_ptr, N, D, s_cn, s_cd, s_ck, s_an, s_ad, s_av, s_gbd, s_gbk, s_gbv, 
                   BD: tl.constexpr, BN: tl.constexpr, BLOCK_N_SIZE: tl.constexpr):
    pid_d = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    rd = pid_d * BD + tl.arange(0, BD)
    rk, rv = tl.arange(0, 4), tl.arange(0, 4)
    acc = tl.zeros((BD, 4, 4), dtype=tl.float32)
    
    n_start = pid_n * BLOCK_N_SIZE
    n_end = tl.minimum(n_start + BLOCK_N_SIZE, N)
    
    for n_st in range(n_start, n_end, BN):
        rn = n_st + tl.arange(0, BN)
        mask_n = rn[:, None, None] < n_end
        mask_d = rd[None, :, None] < D
        gc = tl.load(gc_ptr + (rn[:, None, None]*s_cn + rd[None, :, None]*s_cd + rk[None, None, :]*s_ck), 
                     mask=mask_n & mask_d, other=0.0)
        a = tl.load(a_ptr + (rn[:, None, None]*s_an + rd[None, :, None]*s_ad + rv[None, None, :]*s_av), 
                    mask=mask_n & mask_d, other=0.0)
        acc += tl.sum(gc[:, :, :, None] * a[:, :, None, :], axis=0)

    gb_offsets = (rd[:, None, None]*s_gbd + rk[None, :, None]*s_gbk + rv[None, None, :]*s_gbv)
    mask_gb = (rd[:, None, None] < D)
    tl.atomic_add(gb_ptr + gb_offsets, acc, mask=mask_gb)

class TritonEinsumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        N, D, _ = a.shape
        c = torch.empty((N, D, 4), device=a.device, dtype=a.dtype)
        grid = (triton.cdiv(N, 32), triton.cdiv(D, 32))
        _forward_kernel_TritonEinsum[grid](a, b, c, N, D, *a.stride(), *b.stride(), *c.stride(), BN=32, BD=32)
        ctx.save_for_backward(a, b)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        N, D, _ = a.shape
        grad_a = torch.empty_like(a)
        grad_b = torch.zeros_like(b)
        
        def grid_a(META):
            return (triton.cdiv(N, META['BN']), triton.cdiv(D, META['BD']))
        _grad_a_kernel_TritonEinsum[grid_a](
            grad_output, b, grad_a, N, D, *grad_output.stride(), *b.stride(), *grad_a.stride())
        def grid_b(META):
            return (triton.cdiv(D, META['BD']), triton.cdiv(N, META['BLOCK_N_SIZE']))
        _grad_b_kernel_TritonEinsum[grid_b](
            grad_output, a, grad_b, N, D, *grad_output.stride(), *a.stride(), *grad_b.stride())        
        return grad_a, grad_b
    
    
    
def _build_new_weight_ShortConvFn(weight: torch.Tensor, kernel_size: int,
                       value_channels: int, causal: bool) -> torch.Tensor:
    d = weight.shape[0]
    dv = value_channels
    ks = kernel_size
    pad_total = ks - 1
    pad_left = pad_total if causal else pad_total // 2

    M_padded = F.pad(weight, (pad_total, pad_total))  # (d, ks + 2*pad_total)
    b_idx = torch.arange(pad_total + pad_left, pad_left - 1, -1, device=weight.device)
    # new_weight: (d, len(b_idx), ks) where len(b_idx) == dv == ks (when ks==dv)
    new_weight = M_padded.unfold(dimension=-1, size=ks, step=1)[:, b_idx, :]
    # shape: (d, dv, ks)
    assert new_weight.shape == (d, dv, ks), f"{new_weight.shape}"
    return new_weight


def _build_w_eff_ShortConvFn(weight: torch.Tensor, read: torch.Tensor,
                  kernel_size: int, value_channels: int, causal: bool) -> torch.Tensor:
    new_weight = _build_new_weight_ShortConvFn(weight, kernel_size, value_channels, causal)
    w_eff = (new_weight * read[None, :, None]).sum(dim=1)  # (d, dv)
    return w_eff.contiguous()

@triton.jit
def _fwd_kernel_ShortConvFn(
    X_ptr,       # (BT, d, 4)  row-major
    W_ptr,       # (d, 4)
    OUT_ptr,     # (BT, d)
    BT: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bt = tl.program_id(0)
    pid_d  = tl.program_id(1)

    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask   = d_offs < D

    # Load W_eff: (BLOCK_D, 4)
    w0 = tl.load(W_ptr + d_offs * 4 + 0, mask=mask, other=0.0)
    w1 = tl.load(W_ptr + d_offs * 4 + 1, mask=mask, other=0.0)
    w2 = tl.load(W_ptr + d_offs * 4 + 2, mask=mask, other=0.0)
    w3 = tl.load(W_ptr + d_offs * 4 + 3, mask=mask, other=0.0)

    # Load x: (BLOCK_D, 4)
    base = pid_bt * D * 4 + d_offs * 4
    x0 = tl.load(X_ptr + base + 0, mask=mask, other=0.0)
    x1 = tl.load(X_ptr + base + 1, mask=mask, other=0.0)
    x2 = tl.load(X_ptr + base + 2, mask=mask, other=0.0)
    x3 = tl.load(X_ptr + base + 3, mask=mask, other=0.0)

    out = x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3
    tl.store(OUT_ptr + pid_bt * D + d_offs, out, mask=mask)


class _ShortConvFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_flat, w_eff):
        # x_flat: (BT, D, 4)  w_eff: (D, 4)
        BT, D, _ = x_flat.shape
        out = torch.empty(BT, D, dtype=x_flat.dtype, device=x_flat.device)
        BLOCK_D = 128
        grid = (BT, triton.cdiv(D, BLOCK_D))
        _fwd_kernel_ShortConvFn[grid](
            x_flat, w_eff, out,
            BT=BT, D=D, BLOCK_D=BLOCK_D,
        )
        ctx.save_for_backward(x_flat, w_eff)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        # grad_out: (BT, D)
        x_flat, w_eff = ctx.saved_tensors
        # dx_flat[bt,d,v] = grad_out[bt,d] * w_eff[d,v]
        dx_flat = grad_out.unsqueeze(-1) * w_eff.unsqueeze(0)  # (BT,D,4)
        # dw_eff[d,v] = sum_bt grad_out[bt,d] * x_flat[bt,d,v]
        dw_eff = (grad_out.unsqueeze(-1) * x_flat).sum(0)      # (D,4)
        return dx_flat, dw_eff



@triton.jit
def fwd_delta_kernel(
    X_ptr, K_ptr, V_in_ptr, C_in_ptr,
    W_v_ptr, B_v_ptr, W_beta_ptr, B_beta_ptr,
    X_new_ptr, Khat_ptr, Delta_v_ptr, Inv_norm_ptr, V_ptr, Beta_ptr,
    N, BLOCK_D: tl.constexpr, BLOCK_DV: tl.constexpr,
    EPS: tl.constexpr, V_SCALE: tl.constexpr
):
    pid = tl.program_id(0)
    if pid >= N:
        return

    off_d = tl.arange(0, BLOCK_D)
    off_dv = tl.arange(0, BLOCK_DV)

    # Calculate pointers
    x_ptrs = X_ptr + pid * BLOCK_D * BLOCK_DV + off_d[:, None] * BLOCK_DV + off_dv[None, :]
    k_ptrs = K_ptr + pid * BLOCK_D + off_d
    v_in_ptrs = V_in_ptr + pid * BLOCK_D + off_d
    c_in_ptrs = C_in_ptr + pid * BLOCK_D + off_d

    # 1. Load K and compute normalized k_hat
    k = tl.load(k_ptrs).to(tl.float32)
    norm_sq = tl.sum(k * k) + EPS * EPS
    inv_norm = 1.0 / tl.sqrt(norm_sq)
    k_hat = k * inv_norm
    tl.store(Inv_norm_ptr + pid, inv_norm)
    tl.store(Khat_ptr + pid * BLOCK_D + off_d, k_hat)

    # 2. Compute Beta (fusing nn.Linear)
    c_in = tl.load(c_in_ptrs).to(tl.float32)
    w_beta = tl.load(W_beta_ptr + off_d).to(tl.float32)
    b_beta = tl.load(B_beta_ptr).to(tl.float32)
    beta_logit = tl.sum(c_in * w_beta) + b_beta
    beta = 2.0 / (1.0 + tl.exp(-beta_logit))
    tl.store(Beta_ptr + pid, beta)

    # 3. Compute V (fusing nn.Linear)
    v_in = tl.load(v_in_ptrs).to(tl.float32)
    w_v_ptrs = W_v_ptr + off_dv[:, None] * BLOCK_D + off_d[None, :]
    w_v = tl.load(w_v_ptrs).to(tl.float32)
    b_v = tl.load(B_v_ptr + off_dv).to(tl.float32)
    v_logits = tl.sum(w_v * v_in[None, :], axis=1) + b_v
    v = (1.0 / (1.0 + tl.exp(-v_logits))) * V_SCALE
    tl.store(V_ptr + pid * BLOCK_DV + off_dv, v)

    # 4. Update X
    x = tl.load(x_ptrs).to(tl.float32)
    proj = tl.sum(k_hat[:, None] * x, axis=0)
    delta_v = v - proj
    tl.store(Delta_v_ptr + pid * BLOCK_DV + off_dv, delta_v)

    update = beta * k_hat[:, None] * delta_v[None, :]
    x_new = x + update
    
    # Store with native implicit casting
    tl.store(X_new_ptr + pid * BLOCK_D * BLOCK_DV + off_d[:, None] * BLOCK_DV + off_dv[None, :], x_new)


@triton.jit
def bwd_delta_kernel(
    dY_ptr, X_ptr, Khat_ptr, Delta_v_ptr, Inv_norm_ptr, V_ptr, Beta_ptr,
    dX_ptr, dK_ptr, dV_logits_ptr, dBeta_logits_ptr,
    N, BLOCK_D: tl.constexpr, BLOCK_DV: tl.constexpr, V_SCALE: tl.constexpr
):
    pid = tl.program_id(0)
    if pid >= N:
        return

    off_d = tl.arange(0, BLOCK_D)
    off_dv = tl.arange(0, BLOCK_DV)

    # Calculate pointers
    dy_ptrs = dY_ptr + pid * BLOCK_D * BLOCK_DV + off_d[:, None] * BLOCK_DV + off_dv[None, :]
    x_ptrs = X_ptr + pid * BLOCK_D * BLOCK_DV + off_d[:, None] * BLOCK_DV + off_dv[None, :]
    khat_ptrs = Khat_ptr + pid * BLOCK_D + off_d

    dy = tl.load(dy_ptrs).to(tl.float32)
    x = tl.load(x_ptrs).to(tl.float32)
    khat = tl.load(khat_ptrs).to(tl.float32)
    delta_v = tl.load(Delta_v_ptr + pid * BLOCK_DV + off_dv).to(tl.float32)
    inv_norm = tl.load(Inv_norm_ptr + pid).to(tl.float32)
    v = tl.load(V_ptr + pid * BLOCK_DV + off_dv).to(tl.float32)
    beta = tl.load(Beta_ptr + pid).to(tl.float32)

    # Gradient calculations perfectly derived mathematically
    dk_hat_proj = tl.sum(dy * khat[:, None], axis=0)

    # dX
    dx = dy - beta * khat[:, None] * dk_hat_proj[None, :]
    tl.store(dX_ptr + pid * BLOCK_D * BLOCK_DV + off_d[:, None] * BLOCK_DV + off_dv[None, :], dx)

    # dV Logits (chain-ruled natively)
    dv = beta * dk_hat_proj
    v_sig = v / V_SCALE
    dv_logits = dv * v_sig * (1.0 - v_sig) * V_SCALE
    tl.store(dV_logits_ptr + pid * BLOCK_DV + off_dv, dv_logits)

    # dBeta Logits (chain-ruled natively)
    dbeta = tl.sum(dk_hat_proj * delta_v)
    beta_sig = beta / 2.0
    dbeta_logits = dbeta * beta_sig * (1.0 - beta_sig) * 2.0
    tl.store(dBeta_logits_ptr + pid, dbeta_logits)

    # dK_hat and dK (L2 Norm backprop)
    dy_dot_delta_v = tl.sum(dy * delta_v[None, :], axis=1)
    x_dot_dk_hat_proj = tl.sum(x * dk_hat_proj[None, :], axis=1)
    dkhat = beta * dy_dot_delta_v - beta * x_dot_dk_hat_proj

    khat_dot_dkhat = tl.sum(khat * dkhat)
    dk = inv_norm * (dkhat - khat * khat_dot_dkhat)
    tl.store(dK_ptr + pid * BLOCK_D + off_d, dk)


class FusedDeepDeltaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k_in, v_in, context, W_v, b_v, W_beta, b_beta, config_k_eps, config_v_scale):
        B, T, D, DV = x.shape
        N = B * T
        BLOCK_D, BLOCK_DV = D, DV # DV==4

        x_flat = x.view(N, D, DV).contiguous()
        k_flat = k_in.view(N, D).contiguous()
        v_in_flat = v_in.view(N, D).contiguous()
        c_in_flat = context.view(N, D).contiguous()

        # Allocate strictly necessary memory maps
        x_new = torch.empty_like(x_flat)
        khat = torch.empty_like(k_flat)
        delta_v = torch.empty((N, DV), device=x.device, dtype=torch.float32)
        inv_norm = torch.empty((N,), device=x.device, dtype=torch.float32)
        v_val = torch.empty((N, DV), device=x.device, dtype=torch.float32)
        beta_val = torch.empty((N,), device=x.device, dtype=torch.float32)

        # Mathematical k_eps correction for RMS to L2 substitution
        eps_adj = config_k_eps * math.sqrt(D)

        fwd_delta_kernel[(N,)](
            x_flat, k_flat, v_in_flat, c_in_flat,
            W_v, b_v, W_beta, b_beta,
            x_new, khat, delta_v, inv_norm, v_val, beta_val,
            N, BLOCK_D, BLOCK_DV,
            EPS=eps_adj, V_SCALE=config_v_scale,
            num_warps=8
        )

        ctx.save_for_backward(x_flat, khat, delta_v, inv_norm, v_val, beta_val, v_in_flat, c_in_flat, W_v, W_beta)
        ctx.BLOCK_D = BLOCK_D
        ctx.BLOCK_DV = BLOCK_DV
        ctx.V_SCALE = config_v_scale
        ctx.B, ctx.T = B, T

        return x_new.view(B, T, D, DV)

    @staticmethod
    def backward(ctx, grad_output):
        x_flat, khat, delta_v, inv_norm, v_val, beta_val, v_in_flat, c_in_flat, W_v, W_beta = ctx.saved_tensors
        N = x_flat.shape[0]

        grad_output_flat = grad_output.view(N, ctx.BLOCK_D, ctx.BLOCK_DV).contiguous()

        dx = torch.empty_like(x_flat)
        dk = torch.empty_like(khat)
        dv_logits = torch.empty((N, ctx.BLOCK_DV), device=x_flat.device, dtype=torch.float32)
        dbeta_logits = torch.empty((N,), device=x_flat.device, dtype=torch.float32)

        bwd_delta_kernel[(N,)](
            grad_output_flat, x_flat, khat, delta_v, inv_norm, v_val, beta_val,
            dx, dk, dv_logits, dbeta_logits,
            N, ctx.BLOCK_D, ctx.BLOCK_DV,
            V_SCALE=ctx.V_SCALE,
            num_warps=8
        )

        # Bulk parameter reductions
        dW_v = torch.matmul(dv_logits.t(), v_in_flat)
        db_v = dv_logits.sum(dim=0)
        dv_in = torch.matmul(dv_logits, W_v).view(ctx.B, ctx.T, ctx.BLOCK_D)

        dbeta_logits_unsqueezed = dbeta_logits.unsqueeze(1)
        dW_beta = torch.matmul(dbeta_logits_unsqueezed.t(), c_in_flat)
        
        # FIXED: Removed the .squeeze(0) to maintain shape [1]
        db_beta = dbeta_logits.sum(dim=0, keepdim=True)
        
        dcontext = torch.matmul(dbeta_logits_unsqueezed, W_beta).view(ctx.B, ctx.T, ctx.BLOCK_D)

        return (
            dx.view(ctx.B, ctx.T, ctx.BLOCK_D, ctx.BLOCK_DV),
            dk.view(ctx.B, ctx.T, ctx.BLOCK_D),
            dv_in, dcontext,
            dW_v, db_v, dW_beta, db_beta,
            None, None
        )