# GDN Decode Kernel 구현 배경지식

## 목차

1. [대회 개요](#1-대회-개요)
2. [Gated Delta Network 이론](#2-gated-delta-network-이론)
3. [커널 스펙 상세](#3-커널-스펙-상세)
4. [Delta Rule 수학적 유도](#4-delta-rule-수학적-유도)
5. [GVA (Grouped Value Attention)](#5-gva-grouped-value-attention)
6. [State Layout: k-last](#6-state-layout-k-last)
7. [Triton 커널 설계](#7-triton-커널-설계)
8. [fla-org/flash-linear-attention 참조 구현](#8-fla-orgflash-linear-attention-참조-구현)
9. [최적화 전략](#9-최적화-전략)
10. [DPS (Destination Passing Style)](#10-dps-destination-passing-style)
11. [벤치마크 및 워크로드](#11-벤치마크-및-워크로드)

---

## 1. 대회 개요

**FlashInfer AI Kernel Generation Contest @ MLSys 2026**

NVIDIA Blackwell B200 GPU에서 LLM 추론에 사용되는 고성능 GPU 커널을 구현하는 대회이다. `flashinfer-bench` 프레임워크로 정확성과 성능을 평가한다.

### 대회 트랙

| 트랙 | 설명 |
|------|------|
| **Fused MoE** | FP8 block-scale Mixture-of-Experts (DeepSeek routing) |
| **Sparse Attention** | DeepSeek paged sparse attention |
| **Gated Delta Net** | Gated Delta Network decode/prefill (Qwen3-Next) |

우리가 구현하는 트랙은 **`gdn_decode_qk4_v8_d128_k_last`** — Qwen3-Next 모델의 linear attention layer에서 사용하는 Gated Delta Net의 single-token decode 커널이다.

### 프로젝트 구조

```
flashinfer-kernel-gen/
├── config.toml                  # 트랙/언어/entry point 설정
├── solution/triton/kernel.py    # 커널 구현 (여기에 코드 작성)
├── scripts/
│   ├── pack_solution.py         # solution.json 생성
│   ├── run_local.py             # 로컬 GPU 벤치마크
│   └── run_modal.py             # Modal B200 클라우드 벤치마크
└── mlsys26-contest/             # 대회 정의 + 워크로드 (read-only submodule)
    ├── definitions/gdn/         # 커널 I/O 스펙 + reference 구현
    └── workloads/gdn/           # 실제 추론 워크로드 (JSONL)
```

---

## 2. Gated Delta Network 이론

### 2.1 Linear Attention의 한계와 Delta Rule

기존의 linear attention은 state update가 단순 누적(`state += outer(k, v)`)이라 새로운 정보가 기존 정보를 덮어쓰지 못한다. **Delta Rule**은 이를 해결하기 위해 "현재 state에서 key가 이미 가리키는 value를 빼고, 새로운 value를 더하는" 방식을 사용한다.

### 2.2 Gated Delta Net 핵심 수식

단일 토큰 decode 시 다음 연산이 수행된다:

```
# Gate 계산
g = exp(-exp(A_log) * softplus(a + dt_bias))     # decay gate ∈ (0, 1)
β = sigmoid(b)                                     # update gate ∈ (0, 1)

# Delta Rule State Update
old_v = k @ S                                      # 현재 state에서 key에 해당하는 value 추출
delta_v = β * (v - old_v)                          # 새 value와 기존 value의 차이에 beta 적용
S_new = g * S + outer(k, delta_v)                  # decay 적용 후 rank-1 update

# Output
output = scale * q @ S_new                         # query로 state 읽기
```

여기서:
- `S`는 recurrent state 행렬 `[K, V]` (수학적 표기, 메모리 layout과 다름)
- `g`는 forget gate — 이전 state를 얼마나 유지할지 결정
- `β`는 update gate — 새 정보를 얼마나 반영할지 결정
- `outer(k, delta_v)`는 rank-1 update — key-value 연관을 state에 기록

### 2.3 Gate 계산 상세

**Decay gate `g`:**
```
x = a + dt_bias                    # a: input-dependent, dt_bias: learnable
softplus(x) = log(1 + exp(x))     # 양수 보장, 수치적으로 x > 20일 때 softplus(x) ≈ x
g = exp(-exp(A_log) * softplus(x))  # A_log: learnable log-scale decay
```

- `A_log`는 학습된 파라미터로, `exp(A_log)`가 decay rate를 결정
- `softplus`는 입력에 따른 가변적 decay strength를 제공
- 최종적으로 `g`는 0과 1 사이의 값으로, 1에 가까우면 state를 많이 유지하고 0에 가까우면 많이 잊음

**Update gate `β`:**
```
β = sigmoid(b)                     # b: input-dependent projection
```

- `β`는 delta (새 value - 기존 value)를 얼마나 반영할지 결정
- `β = 1`이면 완전한 delta update, `β = 0`이면 update 없음

### 2.4 Reference 구현과의 등가 관계

대회에서 제공하는 reference 구현은 아래와 같이 전개된다:

```python
old_state = g * h_state                             # [K, V]
old_v = k @ old_state                               # [V]
new_v = beta * v + (1 - beta) * old_v               # [V]
state_remove = outer(k, old_v)                      # [K, V]
state_update = outer(k, new_v)                      # [K, V]
h_state = old_state - state_remove + state_update   # [K, V]
```

이를 수학적으로 정리하면:

```
h_state = old_state + outer(k, new_v - old_v)
        = old_state + outer(k, β*v + (1-β)*old_v - old_v)
        = old_state + outer(k, β*(v - old_v))
        = g*S + outer(k, β*(v - k@(g*S)))
```

이것이 바로 fla-org 커널에서 사용하는 simplified delta rule이다:

```python
b_h *= exp(b_g)                           # state *= g
b_v -= tl.sum(b_h * b_k[None, :], 1)     # v -= k @ state (= v - old_v)
b_v *= b_beta                             # delta_v = β * (v - old_v)
b_h += b_v[:, None] * b_k[None, :]       # state += outer(delta_v, k)
b_o = tl.sum(b_h * b_q[None, :], 1)      # output = q @ state
```

---

## 3. 커널 스펙 상세

### 3.1 Axes (차원 정의)

| Axis | Type | Value | 설명 |
|------|------|-------|------|
| `batch_size` | var | (가변) | 동시 decode 시퀀스 수 |
| `seq_len` | const | **1** | 단일 토큰 decode |
| `num_q_heads` | const | **4** | Query head 수 (TP=4, 16/4=4) |
| `num_k_heads` | const | **4** | Key head 수 |
| `num_v_heads` | const | **8** | Value head 수 (GVA: q/k보다 2배) |
| `head_size` | const | **128** | Head dimension (K = V = 128) |

### 3.2 Input Tensors

| 이름 | Shape | dtype | 설명 |
|------|-------|-------|------|
| `q` | `[B, 1, 4, 128]` | bf16 | Query |
| `k` | `[B, 1, 4, 128]` | bf16 | Key |
| `v` | `[B, 1, 8, 128]` | bf16 | Value |
| `state` | `[B, 8, 128, 128]` | f32 | Recurrent state (k-last, optional) |
| `A_log` | `[8]` | f32 | Log decay parameter (learnable) |
| `a` | `[B, 1, 8]` | bf16 | Input-dependent decay |
| `dt_bias` | `[8]` | f32 | Decay bias (learnable) |
| `b` | `[B, 1, 8]` | bf16 | Update gate input |
| `scale` | scalar | f32 | `1/sqrt(128) ≈ 0.0884` |

### 3.3 Output Tensors

| 이름 | Shape | dtype | 설명 |
|------|-------|-------|------|
| `output` | `[B, 1, 8, 128]` | bf16 | Attention output |
| `new_state` | `[B, 8, 128, 128]` | f32 | Updated recurrent state |

### 3.4 주요 제약 조건

- `num_v_heads >= num_q_heads` (GVA 모드)
- `num_v_heads % num_q_heads == 0` (균등 분배)
- `num_k_heads == num_q_heads`
- State는 optional — `None`일 경우 zero state로 초기화

---

## 4. Delta Rule 수학적 유도

### 4.1 표기법

```
S ∈ ℝ^{K×V}  : state 행렬 (수학적 notation에서 K가 행, V가 열)
q ∈ ℝ^K      : query 벡터
k ∈ ℝ^K      : key 벡터
v ∈ ℝ^V      : value 벡터
g ∈ ℝ        : decay gate (scalar per head)
β ∈ ℝ        : update gate (scalar per head)
```

### 4.2 전체 수식 (한 head에 대해)

```
# 1단계: Decay
S' = g · S

# 2단계: 현재 state에서 key가 가리키는 value 추출
old_v = k^T · S' = Σ_i k_i · S'_{i,j}   ∈ ℝ^V

# 3단계: Error signal 계산
Δv = β · (v - old_v)                      ∈ ℝ^V

# 4단계: Rank-1 update
S_new = S' + k ⊗ Δv = S' + k · Δv^T      ∈ ℝ^{K×V}

# 5단계: Output 계산
o = scale · q^T · S_new = scale · Σ_i q_i · S_new_{i,j}   ∈ ℝ^V
```

### 4.3 Transposed State에서의 수식 (k-last: [V, K])

메모리에 state가 `[V, K]` (k-last) 형태로 저장되어 있을 때, 레지스터에 `b_h[BV, K]`로 로드한다.

이 경우 `b_h[j, i] = S^T[j, i] = S[i, j]` 관계가 성립한다.

```
# k @ S = S^T @ k (transposed)
old_v[j] = Σ_i b_h[j, i] · k[i]  →  tl.sum(b_h * b_k[None, :], axis=1)

# outer(Δv, k) in transposed space
b_h[j, i] += Δv[j] · k[i]       →  b_h += b_v[:, None] * b_k[None, :]

# q @ S = S^T @ q (transposed)
o[j] = Σ_i b_h[j, i] · q[i]     →  tl.sum(b_h * b_q[None, :], axis=1)
```

---

## 5. GVA (Grouped Value Attention)

### 5.1 개념

GVA는 value head 수가 query/key head 수보다 많은 구성이다:

```
num_q_heads = num_k_heads = 4
num_v_heads = 8
ratio = num_v_heads / num_q_heads = 2
```

하나의 q/k head가 **2개의 v head**를 담당한다:

```
v_head 0, 1 → q/k head 0
v_head 2, 3 → q/k head 1
v_head 4, 5 → q/k head 2
v_head 6, 7 → q/k head 3
```

### 5.2 커널에서의 처리

`repeat_interleave`를 사용하지 않고, 커널 내부에서 **integer division으로 매핑**한다:

```python
i_hv = i_bh % HV          # v_head index [0..7]
i_h = i_hv // (HV // H)   # q/k head index [0,0,1,1,2,2,3,3]
```

이렇게 하면:
- `q`와 `k`는 `i_h`로 인덱싱 (4개 head)
- `v`, `state`, `output`은 `i_hv`로 인덱싱 (8개 head)
- 추가 메모리 복사 없이 GVA가 자연스럽게 처리됨

---

## 6. State Layout: k-last

### 6.1 k-last의 의미

State shape: `[B, HV, V, K]`

- 마지막 차원이 K (key dimension) → "k-last"
- V 차원이 K 앞에 위치
- 메모리에서 K가 contiguous (stride=1)

```
state[b, hv, v, k]의 메모리 주소:
  base + (b * HV + hv) * V * K + v * K + k
```

### 6.2 왜 k-last를 사용하는가

Qwen3-Next 모델이 이 layout을 사용하도록 설계되었다. fla-org에서는 `TRANSPOSE_STATE` 플래그로 두 가지 layout을 모두 지원한다:

| Layout | State Shape | 커널 내 레지스터 | 수학적 대응 |
|--------|-------------|-----------------|------------|
| k-first (default) | `[N, HV, K, V]` | `b_h[BK, BV]` | `S[K, V]` |
| k-last (transposed) | `[N, HV, V, K]` | `b_h[BV, BK]` | `S^T[V, K]` |

### 6.3 k-last에서의 메모리 접근 패턴

State를 `[BV, K]` 타일로 로드할 때:
- 각 행 (V 인덱스)에서 K개의 연속된 float32 값을 읽음
- K=128 → 512 bytes/행 → 4개의 128-byte cache line
- V 방향으로는 K stride만큼 건너뜀
- `tl.load`는 이 패턴을 vectorized load로 처리

---

## 7. Triton 커널 설계

### 7.1 Grid 구조

```python
grid = (V // BV, B * HV)
#        ↑          ↑
#    V-tile 수    batch × v_head 수
```

각 program instance가 하나의 `(batch, v_head, V-tile)` 조합을 처리한다.

### 7.2 Block Size 선택

| 파라미터 | 값 | 이유 |
|---------|-----|------|
| BK | K = 128 | K 전체를 하나의 block으로 (NK=1) |
| BV | autotune {8, 16, 32, 64, 128} | V-tile 크기 — 레지스터 압력과 parallelism 트레이드오프 |

**BV에 따른 grid 크기 (B=1, HV=8):**

| BV | Grid | 총 block 수 | B200 SM 대비 |
|----|------|------------|-------------|
| 8 | (16, 8) | 128 | 128/192 = 67% |
| 16 | (8, 8) | 64 | 33% |
| 32 | (4, 8) | 32 | 17% |
| 64 | (2, 8) | 16 | 8% |
| 128 | (1, 8) | 8 | 4% |

### 7.3 레지스터 사용량

State 타일 `b_h[BV, K]`의 레지스터 사용:

| BV | 요소 수 | f32 레지스터 | warps=4 (128 threads) 기준 thread당 |
|----|---------|-------------|-----------------------------------|
| 8 | 1,024 | 4 KB | 8 regs/thread |
| 16 | 2,048 | 8 KB | 16 regs/thread |
| 32 | 4,096 | 16 KB | 32 regs/thread |
| 64 | 8,192 | 32 KB | 64 regs/thread |
| 128 | 16,384 | 64 KB | 128 regs/thread |

GPU thread 당 최대 255 레지스터를 고려하면, BV=128 + 보조 변수도 255 이내에 들어온다.

### 7.4 T=1 Decode의 특수성

일반적인 fused_recurrent 커널은 시간 축(T)에 대한 loop를 포함하지만, decode에서는 T=1이므로:

- Loop body가 **단 1회** 실행
- Loop pipelining (num_stages) 효과 없음 → `num_stages=1` 사용
- 커널이 순수하게 **memory bandwidth bound** — state 읽기/쓰기가 지배적

### 7.5 Autotuning 설정

```python
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
```

- `key=["K", "V"]`: K와 V가 변하면 재-autotune (실제로는 128 고정이므로 1회만 실행)
- warmup=3 iteration 동안 autotune이 완료되므로 실제 벤치마크에 영향 없음

---

## 8. fla-org/flash-linear-attention 참조 구현

### 8.1 개요

`fla-org/flash-linear-attention`은 Songlin Yang, Yu Zhang가 개발한 flash linear attention 커널 라이브러리로, sglang, vllm, TensorRT-LLM, FlagGems 등 주요 LLM 추론 프레임워크에서 채택하고 있다.

**핵심 파일:**
```
fla/ops/gated_delta_rule/
├── fused_recurrent.py    # ← decode 커널 (우리 구현의 기반)
├── chunk.py              # prefill/training용 chunk 커널
└── wy_fast.py            # WY representation 커널
```

### 8.2 fused_recurrent 커널 구조

```python
@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_gated_delta_rule_fwd_kernel(
    q, k, v, g, gk, gv, beta, o, h0, ht, cu_seqlens, scale,
    T, H, HV, K, V, BK, BV,
    ...
):
```

**핵심 설계 결정:**
- `BK = triton.next_power_of_2(K)` — K 전체를 하나의 block
- `BV = min(8, triton.next_power_of_2(V))` — V는 작은 tile (gv 없을 때)
- `num_warps=1, num_stages=3` — 기본 설정
- `grid = (NV, N * HV)` — 2D grid

### 8.3 Gate 전처리 차이

fla-org에서는 gate `g`가 **log-space로 사전 계산**되어 커널에 전달된다:
```python
# 커널 외부 (model layer)
g = -exp(A_log) * softplus(a + dt_bias)   # log-space gate

# 커널 내부
b_h *= exp(b_g)                           # exp를 적용하여 실제 gate 값으로 변환
```

대회 커널에서는 **raw parameter**를 직접 받으므로 커널 내부에서 전체 gate 계산을 수행:
```python
# 커널 내부에서 직접 계산
b_x = b_a_val + b_dt_bias
b_sp = tl.where(b_x > 20.0, b_x, tl.log(1.0 + tl.exp(b_x)))  # softplus
b_log_g = -tl.exp(b_A_log) * b_sp                               # log(g)
b_h *= tl.exp(b_log_g)                                          # state *= g
```

### 8.4 Beta 처리

fla-org에서는 `IS_BETA_HEADWISE` 플래그로 beta가 scalar/per-element인지 구분:

| fla-org 최신 버전 | 의미 |
|-------------------|------|
| `IS_BETA_HEADWISE=True` | beta가 head당 scalar `[B, T, HV]` |
| `IS_BETA_HEADWISE=False` | beta가 element별 `[B, T, HV, V]` |

대회에서는 `b`가 `[B, 1, HV]` shape이므로 **head당 scalar beta**이다.

### 8.5 TRANSPOSE_STATE 지원

fla-org 최신 버전은 `TRANSPOSE_STATE` constexpr로 두 가지 state layout을 지원:

```python
# TRANSPOSE_STATE=False (default): state [K, V]
b_h = tl.zeros([BK, BV], dtype=tl.float32)
b_v -= tl.sum(b_h * b_k[:, None], 0)      # k@state
b_h += b_k[:, None] * b_v[None, :]        # outer(k, Δv)
b_o = tl.sum(b_h * b_q[:, None], 0)       # q@state

# TRANSPOSE_STATE=True: state [V, K]
b_h = tl.zeros([BV, BK], dtype=tl.float32)
b_v -= tl.sum(b_h * b_k[None, :], 1)      # k@state (transposed)
b_h += b_v[:, None] * b_k[None, :]        # outer(Δv, k) (transposed)
b_o = tl.sum(b_h * b_q[None, :], 1)       # q@state (transposed)
```

대회 state는 k-last `[V, K]`이므로 **TRANSPOSE_STATE=True 패턴**을 사용한다.

---

## 9. 최적화 전략

### 9.1 Memory Bandwidth 분석

B=1 decode에서의 메모리 트래픽:

| 데이터 | 크기 | Read/Write |
|--------|------|-----------|
| State (8 heads × 128 × 128 × 4B) | 512 KB | Read + Write = 1 MB |
| q, k (2 × 4 × 128 × 2B) | 2 KB | Read |
| v (8 × 128 × 2B) | 2 KB | Read |
| output (8 × 128 × 2B) | 2 KB | Write |
| 파라미터 (A_log, a, dt_bias, b, scale) | ~100 B | Read |
| **합계** | **~1 MB** | |

B200 HBM 대역폭 ~8 TB/s 기준: 이론적 최소 시간 ≈ **0.125 µs**

실제로는 커널 launch overhead (~5 µs)가 지배적이므로, **block 수를 줄이는 것**이 유리할 수 있다.

### 9.2 Numerically Stable Softplus

```python
# 기본 softplus (overflow 위험)
softplus(x) = log(1 + exp(x))

# 안정적 버전
softplus(x) = x              if x > 20   (exp(x) ≫ 1이므로 log(1+exp(x)) ≈ x)
            = log(1 + exp(x)) otherwise
```

Triton에서:
```python
b_sp = tl.where(b_x > 20.0, b_x, tl.log(1.0 + tl.exp(b_x)))
```

### 9.3 fla-org의 추가 최적화 (고급)

**PTX Inline Assembly Softplus:**
```python
_PTX_SOFTPLUS = """
    .reg .pred p;
    setp.gt.f32  p, ${in_reg}, 20.;
    @p  mov.f32  ${out_reg}, ${in_reg};
    @!p mul.f32            ${out_reg}, ${in_reg}, 1.4426950408889634;  # x * log2(e)
    @!p ex2.approx.ftz.f32 ${out_reg}, ${out_reg};                    # 2^(x*log2(e)) = e^x
    @!p add.f32            ${out_reg}, ${out_reg}, 1.0;                # 1 + e^x
    @!p lg2.approx.ftz.f32 ${out_reg}, ${out_reg};                    # log2(1+e^x)
    @!p mul.f32            ${out_reg}, ${out_reg}, 0.6931471805599453;  # * ln(2)
"""
```

`ex2.approx.ftz.f32` (2^x 근사)와 `lg2.approx.ftz.f32` (log2 근사)는 single PTX instruction으로 처리되어 `tl.exp`/`tl.log`보다 빠를 수 있다. 단, scalar 연산이므로 전체 성능에 미치는 영향은 미미하다.

**exp2 + log2 Space Gates:**
```python
# tl.exp 대신 tl.math.exp2 사용 (PTX 1-instruction)
# gate를 log2 space로 변환하여 연산
```

### 9.4 tl.make_block_ptr (TMA 호환 로드)

Hopper/Blackwell GPU에서는 `tl.make_block_ptr`이 TMA (Tensor Memory Accelerator) 호환 로드를 생성할 수 있다:

```python
p_h = tl.make_block_ptr(
    state_base,          # base pointer
    (V, K),              # 전체 tensor shape
    (K, 1),              # strides (row-major)
    (i_v * BV, 0),       # block offset
    (BV, K),             # block shape
    (1, 0),              # order (K가 contiguous)
)
b_h = tl.load(p_h, boundary_check=(0, 1))
```

현재 구현에서는 pointer arithmetic + mask 방식을 사용하지만, 추가 최적화로 `tl.make_block_ptr`를 고려할 수 있다.

### 9.5 K-tile 분할 (64+64)

fla-org의 chunk kernel은 K=128을 64+64로 분할한다:

```python
b_h1 = tl.zeros([64, BV], dtype=tl.float32)   # K[0:64]
b_h2 = tl.zeros([64, BV], dtype=tl.float32)   # K[64:128]

# k @ state (두 타일로 분할 계산)
kh = tl.sum(b_h1 * b_k1[:, None], 0) + tl.sum(b_h2 * b_k2[:, None], 0)
```

이는 `tl.dot`의 최소 차원 제약(16 이상) 때문이지만, decode (T=1)에서는 element-wise 연산을 사용하므로 K 분할이 필요 없다.

---

## 10. DPS (Destination Passing Style)

### 10.1 개념

DPS는 함수가 결과를 반환하는 대신, **미리 할당된 출력 버퍼에 직접 기록**하는 호출 규약이다:

```python
# Value-returning style (비 DPS)
output, new_state = kernel(q, k, v, state, ...)

# DPS style (대회 기본)
kernel(q, k, v, state, ..., output, new_state)  # output, new_state에 직접 기록
```

### 10.2 DPS의 장점

- Tensor 할당 오버헤드가 벤치마크에 포함되지 않음
- 더 정확한 커널 성능 측정
- 프레임워크에서 메모리 관리를 최적화할 수 있음

### 10.3 함수 시그니처 규칙

**입력 순서**: definition JSON의 `inputs` 키 순서대로
**출력 순서**: definition JSON의 `outputs` 키 순서대로

```python
def kernel(
    # inputs (정의 순서)
    q, k, v, state, A_log, a, dt_bias, b, scale,
    # outputs (정의 순서, pre-allocated)
    output, new_state
):
```

### 10.4 주의사항

- `destination_passing_style: true`가 기본값 → output 파라미터 수를 포함한 총 파라미터 수가 맞아야 함
- 파라미터 수 불일치 시 `expected xx parameters, but got xx` 에러 발생
- Value-returning style을 사용하려면 solution spec에 `destination_passing_style: false` 설정 필요
- **Variadic 인수 사용 금지** — builder validation 실패

---

## 11. 벤치마크 및 워크로드

### 11.1 워크로드 분석

대회에서 제공하는 GDN decode 워크로드 (20개):

| 항목 | 값 |
|------|-----|
| 총 워크로드 수 | 20 |
| batch_size | **모두 1** |
| scale | 0.08838834764831843 (= 1/√128) |
| q, k, v | random 생성 |
| A_log, a, dt_bias, b | safetensors 파일에서 로드 (실제 모델 파라미터) |
| state | random 생성 |

**핵심 관찰**: 모든 워크로드에서 `batch_size=1`이다. 이는 실제 LLM 추론의 decode phase에서 일반적인 설정이다.

### 11.2 벤치마크 설정

```
warmup = 3 iterations      # autotune 완료에 충분
iterations = 100            # 시간 측정
trials = 5                  # 반복 측정으로 분산 감소
GPU = NVIDIA B200 (1×)
timeout = 3600s
```

### 11.3 실행 방법

```bash
# 환경 설정
conda create -n fi-bench python=3.12
conda activate fi-bench
pip install flashinfer-bench modal

# 데이터셋
export FIB_DATASET_PATH=/path/to/mlsys26-contest

# 솔루션 패킹
python scripts/pack_solution.py

# 로컬 벤치마크
python scripts/run_local.py

# B200 클라우드 벤치마크
modal setup
modal volume create flashinfer-trace
modal run scripts/run_modal.py
```

### 11.4 성능 분석 도구

```python
from flashinfer_bench.agents import flashinfer_bench_run_sanitizer, flashinfer_bench_run_ncu

# Memory sanitizer (memcheck, racecheck, synccheck, initcheck)
output = flashinfer_bench_run_sanitizer(solution, workload, sanitizer_types=[...])

# NCU profiling
output = flashinfer_bench_run_ncu(solution, workload, set="detailed", page="details")
```

---

## 부록: 주요 참조 자료

### 코드 참조

| 소스 | 파일/URL | 용도 |
|------|---------|------|
| fla-org | `fla/ops/gated_delta_rule/fused_recurrent.py` | 원본 fused recurrent 커널 |
| sglang | `srt/layers/attention/fla/fused_recurrent.py` | sglang 적용 버전 |
| vllm | `model_executor/layers/fla/ops/kda.py` | vllm 적용 버전 |
| TensorRT-LLM | `_torch/modules/fla/fused_recurrent.py` | TRT-LLM 적용 버전 |
| FlagGems | `fused/FLA/fused_recurrent.py` | FlagGems 적용 버전 |
| 대회 정의 | `mlsys26-contest/definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json` | I/O 스펙 + reference |

### 핵심 개념 요약

| 개념 | 한줄 설명 |
|------|----------|
| Delta Rule | `state += outer(k, β(v - k@state))` — 기존 value를 빼고 새 value를 넣는 associative memory update |
| GVA | Value head가 query/key head보다 많은 구성 — integer division으로 head 매핑 |
| k-last | State shape `[V, K]` — K가 contiguous dimension |
| DPS | 출력 버퍼를 미리 할당하여 함수 인자로 전달하는 호출 규약 |
| Fused Recurrent | Gate 계산 + state update + output을 하나의 커널로 통합 |
| BV Tiling | State의 V 차원을 BV 크기로 분할 — 레지스터 압력과 parallelism 균형 |
