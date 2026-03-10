"""
Optimized GDN Decode Triton Kernel for FlashInfer MLSys 2026 Contest.

Gated Delta Net single-token decode with GVA configuration and k-last state layout.
Adapted from fla-org/flash-linear-attention fused_recurrent_gated_delta_rule.

Track: gdn_decode_qk4_v8_d128_k_last
  - num_q_heads = num_k_heads = 4
  - num_v_heads = 8 (GVA: 2x grouped)
  - head_size = 128 (both K and V)
  - seq_len = 1 (single-token decode)
  - State layout: k-last [B, HV, V, K]

Algorithm (delta rule update per head):
  g = exp(-exp(A_log) * softplus(a + dt_bias))     # decay gate
  beta = sigmoid(b)                                  # update gate
  state = g * state                                  # apply decay
  delta_v = beta * (v - k @ state)                   # error signal
  state = state + outer(delta_v, k)                  # rank-1 update
  output = scale * state @ q                         # read output
"""

import math

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BV": 8}, num_warps=1, num_stages=1),
        triton.Config({"BV": 8}, num_warps=2, num_stages=1),
        triton.Config({"BV": 16}, num_warps=1, num_stages=1),
        triton.Config({"BV": 16}, num_warps=2, num_stages=1),
        triton.Config({"BV": 32}, num_warps=2, num_stages=1),
        triton.Config({"BV": 32}, num_warps=4, num_stages=1),
        triton.Config({"BV": 64}, num_warps=4, num_stages=1),
        triton.Config({"BV": 64}, num_warps=8, num_stages=1),
        triton.Config({"BV": 128}, num_warps=4, num_stages=1),
        triton.Config({"BV": 128}, num_warps=8, num_stages=1),
    ],
    key=["K", "V"],
)
@triton.jit
def gdn_decode_fused_kernel(
    # Inputs
    q_ptr,
    k_ptr,
    v_ptr,
    state_ptr,
    A_log_ptr,
    a_ptr,
    dt_bias_ptr,
    b_ptr,
    scale,
    # Outputs (DPS)
    output_ptr,
    new_state_ptr,
    # Dimensions (constexpr for compile-time optimization)
    H: tl.constexpr,  # num_q_heads = num_k_heads = 4
    HV: tl.constexpr,  # num_v_heads = 8
    K: tl.constexpr,  # head_size = 128
    V: tl.constexpr,  # head_size = 128
    BV: tl.constexpr,  # V block size (autotuned)
    USE_INITIAL_STATE: tl.constexpr,
):
    """
    Fused GDN decode kernel: gate computation + delta rule update + output.

    Grid: (V // BV, B * HV)
    Each program processes one (batch, v_head, V-tile) combination.
    State: k-last layout [V, K] loaded as [BV, K] tiles in registers.
    """
    # ── Program identification ───────────────────────────────────────────
    i_v = tl.program_id(0)  # V-tile index
    i_bh = tl.program_id(1)  # batch * v_head flattened index
    i_b = i_bh // HV  # batch index
    i_hv = i_bh % HV  # v_head index [0..7]
    i_h = i_hv // (HV // H)  # GVA: map v_head → q/k head [0,0,1,1,2,2,3,3]

    # ── Index ranges ─────────────────────────────────────────────────────
    o_k = tl.arange(0, K)  # [0..127] full K dimension
    o_v = i_v * BV + tl.arange(0, BV)  # V-tile indices
    mask_v = o_v < V

    # ── Load state tile [BV, K] ──────────────────────────────────────────
    # state: [B, HV, V, K] contiguous, k-last layout
    # Each tile is a [BV, K] slice where rows=V, cols=K
    p_state = (
        state_ptr + (i_b * HV + i_hv) * V * K + o_v[:, None] * K + o_k[None, :]
    )
    if USE_INITIAL_STATE:
        b_h = tl.load(p_state, mask=mask_v[:, None], other=0.0).to(tl.float32)
    else:
        b_h = tl.zeros([BV, K], dtype=tl.float32)

    # ── Load q[K], k[K] (GVA: indexed by q/k head) ──────────────────────
    # q, k: [B, 1, H, K] contiguous — skip seq_len=1 dim
    b_q = tl.load(q_ptr + (i_b * H + i_h) * K + o_k).to(tl.float32)
    b_k = tl.load(k_ptr + (i_b * H + i_h) * K + o_k).to(tl.float32)

    # ── Load v[BV] ──────────────────────────────────────────────────────
    # v: [B, 1, HV, V] contiguous
    b_v = tl.load(
        v_ptr + (i_b * HV + i_hv) * V + o_v, mask=mask_v, other=0.0
    ).to(tl.float32)

    # ── Compute gate g and beta from raw parameters ──────────────────────
    # All are scalar per (batch, v_head)
    b_A_log = tl.load(A_log_ptr + i_hv).to(tl.float32)
    b_dt_bias = tl.load(dt_bias_ptr + i_hv).to(tl.float32)
    b_a_val = tl.load(a_ptr + i_b * HV + i_hv).to(tl.float32)  # a: [B, 1, HV]
    b_b_val = tl.load(b_ptr + i_b * HV + i_hv).to(tl.float32)  # b: [B, 1, HV]

    # softplus(a + dt_bias) — numerically stable
    b_x = b_a_val + b_dt_bias
    b_sp = tl.where(b_x > 20.0, b_x, tl.log(1.0 + tl.exp(b_x)))

    # log(g) = -exp(A_log) * softplus(a + dt_bias)
    # g = exp(log_g) ∈ (0, 1)
    b_log_g = -tl.exp(b_A_log) * b_sp

    # beta = sigmoid(b) ∈ (0, 1)
    b_beta = tl.sigmoid(b_b_val)

    # Pre-scale q (saves one multiply in the output computation)
    b_q = b_q * scale

    # ── Delta Rule Update (k-last state [BV, K]) ────────────────────────
    #
    # In k-last layout, b_h[j, i] represents state^T[v_j, k_i]
    # All operations use transposed formulas:
    #   k @ state = sum_k state^T[v, k] * k[k]  →  tl.sum(b_h * b_k, axis=1)
    #   outer(v, k) in transposed space           →  v[:, None] * k[None, :]
    #   q @ state = sum_k state^T[v, k] * q[k]  →  tl.sum(b_h * b_q, axis=1)

    # 1. Apply decay: state *= exp(log_g) = g
    b_h *= tl.exp(b_log_g)

    # 2. Compute delta: v -= k @ state  (error signal)
    b_v -= tl.sum(b_h * b_k[None, :], 1)

    # 3. Apply beta gate
    b_v *= b_beta

    # 4. Rank-1 state update: state += delta_v ⊗ k
    b_h += b_v[:, None] * b_k[None, :]

    # 5. Compute output: o = scale * q @ state (q already pre-scaled)
    b_o = tl.sum(b_h * b_q[None, :], 1)

    # ── Store output [B, 1, HV, V] ──────────────────────────────────────
    tl.store(
        output_ptr + (i_b * HV + i_hv) * V + o_v,
        b_o.to(tl.bfloat16),
        mask=mask_v,
    )

    # ── Store updated state [B, HV, V, K] ───────────────────────────────
    p_new_state = (
        new_state_ptr
        + (i_b * HV + i_hv) * V * K
        + o_v[:, None] * K
        + o_k[None, :]
    )
    tl.store(p_new_state, b_h.to(tl.float32), mask=mask_v[:, None])


def kernel(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state):
    """
    GDN decode kernel entry point (Destination Passing Style).

    Inputs:
        q:       [B, 1, 4, 128]    bfloat16   Query tensor
        k:       [B, 1, 4, 128]    bfloat16   Key tensor
        v:       [B, 1, 8, 128]    bfloat16   Value tensor
        state:   [B, 8, 128, 128]  float32    Recurrent state (k-last, optional)
        A_log:   [8]               float32    Log decay parameter
        a:       [B, 1, 8]         bfloat16   Input-dependent decay
        dt_bias: [8]               float32    Decay bias
        b:       [B, 1, 8]         bfloat16   Update gate input
        scale:   float32                      Scale factor (1/sqrt(head_size))

    Outputs (pre-allocated):
        output:    [B, 1, 8, 128]    bfloat16   Attention output
        new_state: [B, 8, 128, 128]  float32    Updated recurrent state
    """
    B = q.shape[0]
    H = q.shape[2]  # num_q_heads = num_k_heads = 4
    HV = v.shape[2]  # num_v_heads = 8
    K = q.shape[3]  # head_size = 128
    V = v.shape[3]  # head_size = 128

    # Handle scale default
    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(K)
    if isinstance(scale, torch.Tensor):
        scale = scale.item()

    # Handle optional initial state
    if state is None:
        state = torch.zeros_like(new_state)

    # Ensure contiguous memory layout for pointer arithmetic
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    state = state.contiguous()
    a = a.contiguous()
    b = b.contiguous()

    # Launch kernel — grid tiles over V dimension and batch*v_heads
    grid = lambda META: (triton.cdiv(V, META["BV"]), B * HV)

    gdn_decode_fused_kernel[grid](
        q,
        k,
        v,
        state,
        A_log,
        a,
        dt_bias,
        b,
        scale,
        output,
        new_state,
        H=H,
        HV=HV,
        K=K,
        V=V,
        USE_INITIAL_STATE=True,  # Always True (we allocate zeros above if None)
    )
