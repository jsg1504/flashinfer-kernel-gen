# Triton GPU Kernel Optimization Skill

You are optimizing a **Gated Delta Network (GDN) decode** Triton kernel for the FlashInfer MLSys 2026 Contest. Target: NVIDIA Blackwell B200 GPU.

## Project Context

- **Track**: `gdn_decode_qk4_v8_d128_k_last`
- **Kernel**: `solution/triton/kernel.py` → entry `kernel.py::kernel`
- **Config**: `config.toml`
- **Spec**: `mlsys26-contest/definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json`
- **Optimization log**: `optimize.md` (applied + candidate list, attempt history)
- **Technique reference**: `know-how.md` (detailed Korean documentation)

## Kernel Characteristics

- **Algorithm**: Delta rule recurrent state update (linear attention variant)
- **Workload**: T=1 single-token decode, batch_size=1, H=4 q/k heads, HV=8 v heads, K=V=128
- **Data flow**: State [B,8,128,128] f32 read → gate + delta update → State write + output [B,1,8,128] bf16
- **Memory**: ~1MB state dominates (512KB read + 512KB write)
- **Bound**: Memory bandwidth bound (at batch_size=1) with kernel launch overhead

## Optimization Decision Framework

### Step 1: Classify the Bottleneck

```
Memory-bound?     → state read/write dominates
  → Optimize: TMA loads, coalescing, data layout, reduce state traffic
  → Tools: tl.make_block_ptr, async copy, persistent kernels

Compute-bound?    → FLOPs dominate (unlikely at batch_size=1)
  → Optimize: PTX intrinsics, exp2/log2, reduce ops count
  → Tools: tl.math.exp2, inline PTX, algorithmic simplification

Launch-overhead?  → kernel launch cost ≈ computation time
  → Optimize: fewer blocks, larger BV, merge kernels
  → Tools: increase BV, reduce grid size
```

### Step 2: Evaluate Candidate Optimizations

Before implementing, answer these questions:
1. **Impact**: Will this meaningfully reduce latency? (>5% improvement threshold)
2. **Risk**: Can this break correctness? (f32→bf16 changes, axis transpositions)
3. **Complexity**: Is the code complexity increase justified?
4. **Measurability**: Can I verify the improvement with the benchmark?

### Step 3: One Change at a Time

- Implement exactly ONE optimization per attempt
- Benchmark before AND after
- If correctness fails → revert immediately (`git checkout -- solution/triton/kernel.py`)
- If perf regresses → revert and document why in optimize.md
- Track every attempt (success or failure) in optimize.md

## Optimization Catalog

### Currently Applied (O1–O15)

| ID | Technique | Category |
|----|-----------|----------|
| O1 | Fused kernel (gate + delta + output) | Algorithm |
| O2 | Delta rule simplification (1 outer product) | Algorithm |
| O3 | Query pre-scaling | Algorithm |
| O4 | In-place register operations | Register |
| O5 | Full K=128 single block | Tiling |
| O6 | V-only tiling with autotune | Tiling |
| O7 | constexpr dimensions | Compile |
| O8 | GVA integer division head mapping | Memory |
| O9 | Numerically stable softplus | Numeric |
| O10 | Log-space gate computation | Numeric |
| O11 | f32 compute / f32 state | Numeric |
| O12 | Coalesced K-contiguous state access | Memory |
| O13 | Scalar broadcast loads for gates | Memory |
| O14 | num_stages=1 (no pipelining for T=1) | Compile |
| O15 | Contiguous tensor enforcement | Memory |

### Candidates (Not Yet Applied)

| ID | Technique | Expected Impact | Risk |
|----|-----------|----------------|------|
| C1 | `tl.make_block_ptr` TMA loads | SM ALU savings, HW boundary check | Low — drop-in replacement |
| C2 | PTX inline softplus (ex2/lg2) | ~1 instruction/head | Very Low — scalar only |
| C3 | `tl.math.exp2` / `tl.math.log2` | 1 PTX instruction exp/log | Very Low — scalar only |
| C4 | Persistent kernel pattern | SM reuse for large batch | N/A — batch=1 only |
| C5 | K-tile split (64+64) | Register pressure relief | Medium — needs sync |
| C6 | Warp specialization | Load/compute overlap | High — complex for T=1 |

## Triton-Specific Patterns

### Memory Access
```python
# Coalesced load (K is contiguous — GOOD)
b_h = tl.load(state_ptr + row * K + tl.arange(0, K))

# TMA block load (Hopper/Blackwell)
p = tl.make_block_ptr(base, shape=(V,K), strides=(K,1), offsets=(off,0), block_shape=(BV,K), order=(1,0))
b_h = tl.load(p, boundary_check=(0,))
```

### Numeric Stability
```python
# Stable softplus: threshold=20 (float32 safe)
sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))

# Log-space gate: compute log(g) then exp once
log_g = -tl.exp(A_log) * sp
state *= tl.exp(log_g)
```

### Autotuning
```python
# BV tiling: trade parallelism vs register pressure
# Small BV (8-16): more blocks, better SM utilization, less register per thread
# Large BV (64-128): fewer blocks, more register per thread, less launch overhead
# num_stages=1: no loop pipelining needed for T=1 decode
```

## Critical Constraints

- **DPS convention**: kernel signature must be `(inputs..., output, new_state)` — never return tensors
- **No variadic args**: all parameters explicitly declared
- **k-last state layout**: `[B, HV, V, K]` — axis directions in tl.sum must match
- **f32 state**: NEVER use bf16 for state — recurrent error accumulation
- **Correctness first**: all 20 workloads must PASS before considering performance
