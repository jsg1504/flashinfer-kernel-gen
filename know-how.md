# GDN Decode Triton 커널 최적화 노하우

> 이 문서는 `gdn_decode_qk4_v8_d128_k_last` Triton 커널을 구현하면서 적용한 최적화 기법과 실전 노하우를 정리한 것이다.
> 이론적 배경은 `background.md`를 참고하라.

---

## 목차

1. [알고리즘 수준 최적화](#1-알고리즘-수준-최적화)
2. [Triton 커널 설계 노하우](#2-triton-커널-설계-노하우)
3. [메모리 접근 최적화](#3-메모리-접근-최적화)
4. [수치 안정성 기법](#4-수치-안정성-기법)
5. [Autotuning 전략](#5-autotuning-전략)
6. [GVA 처리 패턴](#6-gva-처리-패턴)
7. [DPS 호출 규약 실전 가이드](#7-dps-호출-규약-실전-가이드)
8. [흔한 실수와 디버깅](#8-흔한-실수와-디버깅)
9. [추가 최적화 후보 (미적용)](#9-추가-최적화-후보-미적용)
10. [벤치마크 결과 분석](#10-벤치마크-결과-분석)

---

## 1. 알고리즘 수준 최적화

### 1.1 Reference 구현의 중복 연산 제거

대회에서 제공하는 reference 구현은 다음과 같이 **중복 연산**이 존재한다:

```python
# Reference 구현 (비효율적)
old_state = g * h_state                             # ① decay
old_v = k @ old_state                               # ② k@state
new_v = beta * v + (1 - beta) * old_v               # ③ 보간
state_remove = outer(k, old_v)                      # ④ 제거할 부분 (outer)
state_update = outer(k, new_v)                      # ⑤ 추가할 부분 (outer)
h_state = old_state - state_remove + state_update   # ⑥ 최종 state
```

여기서 `state_remove`와 `state_update`에 각각 `outer`를 계산하면 **outer product를 2번** 수행한다.
수학적으로 정리하면 이를 **1번의 outer product**로 줄일 수 있다:

```
h_state = old_state + outer(k, new_v - old_v)
        = old_state + outer(k, β*(v - old_v))
```

**최적화된 6단계 (O17 적용 후 실제 커널):**

```python
b_h *= tl.exp(b_log_g)                    # ① decay (in-place)
b_v -= tl.sum(b_h * b_k[None, :], 1)     # ② delta: v -= k@state
b_v *= b_beta                             # ③ beta gate (in-place)
b_kq = tl.sum(b_k * b_q)                 # ④ scalar: dot(k, q_scaled)
b_o = tl.sum(b_h * b_q[None, :], 1) + b_v * b_kq  # ⑤ output (algebraic identity)
b_h += b_v[:, None] * b_k[None, :]       # ⑥ rank-1 update (now independent of output)
```

**효과:**
- outer product 2회 → 1회 (연산량 절반)
- intermediate tensor 2개 → 0개 (레지스터 절약)
- fla-org/flash-linear-attention에서도 동일한 최적화를 사용

### 1.2 Query Pre-scaling

Output 계산에서 `scale`을 곱하는 위치를 바꿔 연산을 절약:

```python
# Before (매 output element에 scale 곱셈)
b_o = tl.sum(b_h * b_q[None, :], 1) * scale     # BV개의 곱셈

# After (q에 한 번만 scale 적용)
b_q = b_q * scale                                 # K개의 곱셈 (한 번만)
b_o = tl.sum(b_h * b_q[None, :], 1)              # scale 이미 반영됨
```

**효과:**
- `scale` 곱셈 위치를 q 로드 직후로 이동
- BV 크기와 무관하게 K(=128)번의 곱셈으로 고정
- 특히 BV가 큰 config에서 유리

### 1.3 Gate 계산 Fusion

fla-org에서는 gate `g`를 커널 외부에서 사전 계산하여 log-space 값으로 전달한다.
대회에서는 raw parameter(`A_log`, `a`, `dt_bias`, `b`)가 직접 입력되므로, 커널 내부에서 전체 gate 계산을 fuse해야 한다:

```python
# 커널 내부에서 모든 gate 연산을 fuse
b_A_log = tl.load(A_log_ptr + i_hv)        # scalar load
b_dt_bias = tl.load(dt_bias_ptr + i_hv)    # scalar load
b_a_val = tl.load(a_ptr + i_b * HV + i_hv) # scalar load
b_b_val = tl.load(b_ptr + i_b * HV + i_hv) # scalar load

# softplus + decay gate + update gate 전부 fuse
b_x = b_a_val + b_dt_bias
b_sp = tl.where(b_x > 20.0, b_x, tl.log(1.0 + tl.exp(b_x)))
b_log_g = -tl.exp(b_A_log) * b_sp
b_beta = tl.sigmoid(b_b_val)
```

**효과:**
- 별도의 gate 전처리 커널 launch 불필요
- 추가 global memory read/write 제거
- Gate 계산은 scalar 연산 4~5개뿐이라 커널 launch overhead 대비 이득이 큼

### 1.4 Algebraic Output Reformulation (O17)

Output 계산을 updated state가 아닌 decayed state에서 수행하여 critical path를 단축:

```python
# Before (output depends on state update):
b_h += b_v[:, None] * b_k[None, :]       # state update ← output이 이것에 의존
b_o = tl.sum(b_h * b_q[None, :], 1)      # output from UPDATED state

# After (output independent of state update):
b_kq = tl.sum(b_k * b_q)                 # scalar: dot(k, q)
b_o = tl.sum(b_h * b_q[None, :], 1) + b_v * b_kq  # output from DECAYED state + correction
b_h += b_v[:, None] * b_k[None, :]       # state update (now independent)
```

**수학적 증명:**
```
output = q @ (state + outer(delta, k))      # 원래 공식
       = q @ state + q @ outer(delta, k)    # 분배법칙
       = q @ state + delta * (k · q)        # outer product의 contraction
```

**효과:**
- State update와 output 계산 사이의 RAW register dependency 제거
- 컴파일러가 output store와 state update/store를 병렬 스케줄링 가능
- 추가 비용: scalar dot product (K muls + K adds) + BV muls + BV adds — 무시 가능
- Memory-bound 커널에서는 효과 미미하지만, HBM이 빠른 GPU(B200)에서 compute overlap 이점 기대
---

## 2. Triton 커널 설계 노하우

### 2.1 K 차원 전체를 하나의 Block으로

이 커널의 가장 중요한 설계 결정은 **K=128 전체를 하나의 block으로 처리**하는 것이다:

```python
# K를 분할하지 않음 — NK=1
o_k = tl.arange(0, K)          # [0..127] 한 번에
b_q = tl.load(q_ptr + ... + o_k)  # K 전체를 한 번에 로드
```

**왜 K를 분할하지 않는가:**

1. **Delta Rule의 구조적 제약**: `b_v -= tl.sum(b_h * b_k[None, :], axis=1)`에서 K 차원 전체에 대한 reduction이 필요. K를 분할하면 partial sum을 합쳐야 하므로 thread 간 synchronization이 발생.

2. **K=128은 레지스터에 충분히 들어감**: float32 기준 128개 = 512 bytes. State 타일 `b_h[BV, 128]`에서 BV=32일 때도 thread당 32 레지스터만 사용.

3. **fla-org도 동일한 결정**: `BK = triton.next_power_of_2(K)` — K 전체를 하나의 block으로 사용.

### 2.2 V 차원만 Tiling

K를 고정하고 V 차원만 BV 크기로 tiling한다:

```python
grid = lambda META: (triton.cdiv(V, META["BV"]), B * HV)
#                    ^^^^^^^^^^^^^^^^^^^^^^^^    ^^^^^^^
#                    V를 BV 크기로 분할           batch × v_head
```

**V-tiling의 장점:**

- V 차원의 각 tile은 완전히 독립적 — 동기화 불필요
- 각 program instance가 state의 `[BV, K]` 슬라이스를 독립적으로 처리
- BV를 줄이면 block 수가 늘어나 GPU SM을 더 많이 활용

### 2.3 constexpr 적극 활용

Triton에서 `tl.constexpr`로 선언된 파라미터는 **컴파일 타임 상수**가 된다:

```python
def gdn_decode_fused_kernel(
    ...,
    H: tl.constexpr,           # 4 → 컴파일 타임 상수
    HV: tl.constexpr,          # 8 → 컴파일 타임 상수
    K: tl.constexpr,           # 128 → 컴파일 타임 상수
    V: tl.constexpr,           # 128 → 컴파일 타임 상수
    BV: tl.constexpr,          # autotune → 각 config마다 별도 컴파일
    USE_INITIAL_STATE: tl.constexpr,
):
```

**효과:**
- `tl.arange(0, K)` → 컴파일러가 K=128을 알고 최적화된 코드 생성
- `i_h = i_hv // (HV // H)` → `HV // H = 2`가 컴파일 타임에 결정되어 shift 연산으로 대체
- `if USE_INITIAL_STATE:` → dead code elimination으로 사용하지 않는 분기 제거
- 불필요한 runtime 분기와 동적 계산 전부 제거

### 2.4 In-place 연산 최대화

중간 결과를 새 변수에 저장하지 않고, 기존 변수를 in-place로 갱신:

```python
# ✅ In-place (레지스터 재사용)
b_h *= tl.exp(b_log_g)          # state decay: b_h 재사용
b_v -= tl.sum(b_h * b_k[None, :], 1)   # delta: b_v 재사용
b_v *= b_beta                   # beta gate: b_v 재사용
b_h += b_v[:, None] * b_k[None, :]     # update: b_h 재사용

# ❌ Non-in-place (추가 레지스터 필요)
b_h_decayed = b_h * tl.exp(b_log_g)
b_delta = b_v - tl.sum(b_h_decayed * b_k[None, :], 1)
b_delta_gated = b_delta * b_beta
b_h_new = b_h_decayed + b_delta_gated[:, None] * b_k[None, :]
```

**효과:**
- intermediate tensor 할당 제거 → 레지스터 압력 감소
- 컴파일러가 register reuse를 더 쉽게 결정
- 코드가 간결해져 실수 가능성 감소

---

## 3. 메모리 접근 최적화

### 3.1 Coalesced Access 보장

State `[B, HV, V, K]`에서 K가 마지막 차원(contiguous)이므로, **K 방향 로드가 coalesced**:

```python
# State 로드: b_h[BV, K]
# 각 thread가 K 방향으로 연속된 메모리를 읽음
p_state = state_ptr + (i_b * HV + i_hv) * V * K + o_v[:, None] * K + o_k[None, :]
b_h = tl.load(p_state, mask=mask_v[:, None], other=0.0)
```

K=128, float32 → 512 bytes/row → warp 내 32 thread가 각각 4개의 float32(16 bytes)를 담당.
이는 GPU의 128-byte cache line과 잘 정렬된다.

### 3.2 q, k는 Broadcast 로드

q와 k는 `[K]` 크기의 1D 벡터이며, 같은 q/k head를 공유하는 모든 V-tile에서 동일:

```python
b_q = tl.load(q_ptr + (i_b * H + i_h) * K + o_k)   # [K] 벡터
b_k = tl.load(k_ptr + (i_b * H + i_h) * K + o_k)   # [K] 벡터
```

**노하우:**
- GVA에서 같은 q/k head를 사용하는 v_head pair (예: v_head 0,1이 모두 q/k head 0)가 같은 데이터를 L2 cache에서 hit할 수 있음
- `b_q * scale`를 로드 직후에 한 번 하면, 이후 모든 V-tile 계산에서 pre-scaled 값을 사용

### 3.3 Scalar Parameter 로드 최적화

Gate 계산에 필요한 `A_log`, `dt_bias`, `a`, `b`는 모두 **scalar** (per-head 또는 per-batch-head):

```python
# 이 4개는 전부 scalar → register 하나에 들어감
b_A_log = tl.load(A_log_ptr + i_hv)
b_dt_bias = tl.load(dt_bias_ptr + i_hv)
b_a_val = tl.load(a_ptr + i_b * HV + i_hv)
b_b_val = tl.load(b_ptr + i_b * HV + i_hv)
```

**노하우:**
- Scalar 로드는 모든 thread가 같은 주소를 읽으므로 broadcast load로 처리됨
- L1 cache에서 서비스되므로 HBM bandwidth를 거의 소모하지 않음
- 커널 시작 시 한 번 로드하면 이후 재사용

### 3.4 Contiguous 보장

Python wrapper에서 `.contiguous()` 호출로 입력 tensor의 메모리 레이아웃을 보장:

```python
q = q.contiguous()
k = k.contiguous()
v = v.contiguous()
state = state.contiguous()
a = a.contiguous()
b = b.contiguous()
```

**왜 필요한가:**
- PyTorch tensor는 slice, transpose, view 등으로 non-contiguous할 수 있음
- 커널 내부의 pointer arithmetic은 contiguous layout을 가정
- Non-contiguous tensor에서 `tl.load`는 잘못된 데이터를 읽음
- `.contiguous()`는 이미 contiguous인 경우 no-op (비용 없음)

---

## 4. 수치 안정성 기법

### 4.1 Numerically Stable Softplus

Softplus `log(1 + exp(x))`는 `x`가 클 때 `exp(x)`가 overflow한다:

```python
# ❌ Naive (x > 88에서 float32 overflow)
b_sp = tl.log(1.0 + tl.exp(b_x))

# ✅ Stable (x > 20이면 softplus(x) ≈ x, 오차 < 2e-9)
b_sp = tl.where(b_x > 20.0, b_x, tl.log(1.0 + tl.exp(b_x)))
```

**왜 threshold = 20.0인가:**
- `exp(20) ≈ 4.85 × 10^8`
- `log(1 + exp(20)) ≈ 20.000000002` (x와의 오차 < 2e-9)
- float32 정밀도(~7자리)에서 20 이상이면 `softplus(x) = x`와 구분 불가
- fla-org도 동일한 threshold 사용

### 4.2 Log-space Gate 계산

Gate `g`를 직접 계산하지 않고 log-space에서 계산 후 `exp`를 적용:

```python
# log(g) = -exp(A_log) * softplus(a + dt_bias)
b_log_g = -tl.exp(b_A_log) * b_sp

# g = exp(log_g)는 state에 직접 적용
b_h *= tl.exp(b_log_g)   # state *= g
```

**왜 log-space인가:**
- `g`는 (0, 1) 범위의 값으로, 직접 곱하면 underflow 위험
- Log-space에서는 `log_g`가 (-∞, 0) 범위 → underflow 없음
- 여러 timestep 누적 시 log-space에서 덧셈으로 처리 가능 (T=1에서는 해당 없음)
- `tl.exp(b_log_g)`가 0에 가까워도 float32에서 정확하게 표현

### 4.3 Float32 State 유지

State는 반드시 float32로 유지:

```python
b_h = tl.load(p_state, ...).to(tl.float32)   # f32로 로드
# ... 모든 연산 f32 ...
tl.store(p_new_state, b_h.to(tl.float32), ...)  # f32로 저장
```

**왜 bf16이 아닌가:**
- State는 recurrent — 매 decode step마다 갱신되므로 오차가 누적됨
- bf16의 mantissa는 7bit (유효숫자 ~2.4자리) — 수천 step 후 state가 발산 가능
- Output만 bf16으로 cast: `b_o.to(tl.bfloat16)` — 이것은 최종 값이므로 안전

### 4.4 Input bf16 → f32 변환

입력 tensor는 bf16으로 들어오지만, 모든 연산은 f32에서 수행:

```python
b_q = tl.load(q_ptr + ...).to(tl.float32)
b_k = tl.load(k_ptr + ...).to(tl.float32)
b_v = tl.load(v_ptr + ...).to(tl.float32)
```

**효과:**
- 로드 시 bf16 → f32 변환은 하드웨어에서 무비용 (NVIDIA GPU의 type conversion unit)
- Delta rule의 뺄셈 `b_v -= tl.sum(...)` 에서 bf16 정밀도 부족으로 큰 오차 발생 방지
- 참조 구현도 float32 matmul을 사용 — 정밀도 매칭에 필수

---

## 5. Autotuning 전략

### 5.1 Autotune Config 설계 원칙

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

**설계 결정의 근거:**

| 요소 | 선택 | 이유 |
|------|------|------|
| BV 범위 | {8, 16, 32, 64, 128} | 2의 거듭제곱, V=128을 균등 분할 |
| num_warps | BV에 비례하여 증가 | BV가 클수록 thread가 더 많은 데이터 처리 |
| num_stages | 항상 1 | T=1이므로 pipeline 효과 없음 |
| key | ["K", "V"] | K, V가 변하면 재-autotune (실제로는 128 고정) |

### 5.2 BV와 num_warps의 관계

```
BV=8   → 1-2 warps : 작은 tile, thread당 적은 일, warp 수 적게
BV=16  → 1-2 warps : 동일 논리
BV=32  → 2-4 warps : 중간 tile
BV=64  → 4-8 warps : 큰 tile, thread 활용도 높임
BV=128 → 4-8 warps : 최대 tile, 최대 thread
```

**일반 규칙**: `num_warps ≈ BV × K × sizeof(f32) / (warp_size × target_regs_per_thread × sizeof(f32))`

그러나 정확한 최적값은 GPU 아키텍처마다 다르므로 autotune에 맡기는 것이 정답.

### 5.3 num_stages = 1인 이유

`num_stages`는 Triton의 software pipelining 단계 수를 제어한다:

- `num_stages > 1`: loop body의 여러 iteration을 중첩시켜 memory latency를 숨김
- **T=1 decode에서는 loop가 없음** — pipelining 대상이 없으므로 `num_stages=1`이 최적
- `num_stages > 1`로 설정하면 불필요한 prefetch 코드가 생성되어 오히려 느려질 수 있음

### 5.4 key 파라미터

```python
key=["K", "V"]
```

- `key`에 지정된 파라미터가 변할 때마다 autotune을 다시 수행
- `K=128`, `V=128`이 워크로드 전체에서 고정이므로, 실질적으로 **autotune은 1회만** 실행
- warmup=3 iteration 내에 autotune이 완료되어 벤치마크 측정에 영향 없음

---

## 6. GVA 처리 패턴

### 6.1 Integer Division으로 Head 매핑

GVA (num_v_heads=8, num_q_heads=4)에서 v_head → q/k head 매핑:

```python
i_hv = i_bh % HV          # v_head index: 0,1,2,3,4,5,6,7
i_h = i_hv // (HV // H)   # q/k head:    0,0,1,1,2,2,3,3
```

**왜 `repeat_interleave`를 사용하지 않는가:**

```python
# ❌ Python에서 q, k를 확장 (추가 메모리 + 복사)
q_expanded = q.repeat_interleave(HV // H, dim=2)  # [B,1,8,128] — 2배 메모리
k_expanded = k.repeat_interleave(HV // H, dim=2)

# ✅ 커널 내부에서 integer division (추가 비용 0)
i_h = i_hv // 2   # 컴파일 타임에 shift로 대체
```

**효과:**
- 메모리 복사 제거 (q, k 데이터를 2배로 확장하지 않음)
- Global memory bandwidth 절약 (q, k는 원본 크기 그대로)
- `HV // H = 2` → 컴파일 타임 상수이므로 `shr` (shift right) 1개의 instruction으로 처리

### 6.2 Grid에서의 V_head 분리

Grid의 두 번째 차원이 `B * HV`:

```python
grid = lambda META: (triton.cdiv(V, META["BV"]), B * HV)

# 커널 내부에서 분리
i_b = i_bh // HV    # batch index
i_hv = i_bh % HV    # v_head index
```

**왜 `(V//BV, B, HV)` 3D grid를 사용하지 않는가:**
- Triton은 3D grid를 지원하지만, 2D가 simpler하고 index 계산이 명확
- fla-org도 동일하게 `(NV, N * H)` 또는 `(NV, N * HV)` 2D grid 사용
- `B * HV`를 하나의 차원으로 합치면 program_id 하나로 batch와 head를 모두 결정

---

## 7. DPS 호출 규약 실전 가이드

### 7.1 파라미터 순서 규칙

DPS에서 함수 시그니처는 **inputs → outputs** 순서:

```python
def kernel(
    # Inputs (definition JSON의 inputs 키 순서 그대로)
    q, k, v, state, A_log, a, dt_bias, b, scale,
    # Outputs (definition JSON의 outputs 키 순서 그대로)
    output, new_state
):
```

**주의:** 순서를 바꾸면 `flashinfer-bench`가 잘못된 tensor를 전달하여 correctness 실패.

### 7.2 Optional State 처리

State가 `None`일 수 있으므로 wrapper에서 처리:

```python
if state is None:
    state = torch.zeros_like(new_state)    # new_state와 같은 shape/dtype/device
```

**노하우:**
- `torch.zeros_like(new_state)`를 사용하면 shape, dtype, device를 자동으로 맞춤
- 커널에서는 `USE_INITIAL_STATE: tl.constexpr`로 분기했으나, 실제로는 항상 `True`로 설정하고 wrapper에서 zero tensor를 생성하는 것이 simpler
- Zero tensor 할당은 벤치마크에 포함되지만, 실제로 state=None인 경우는 첫 decode step뿐

### 7.3 Scale 처리

```python
if scale is None or scale == 0.0:
    scale = 1.0 / math.sqrt(K)
if isinstance(scale, torch.Tensor):
    scale = scale.item()      # Tensor → Python float
```

**왜 `.item()` 호출이 필요한가:**
- `flashinfer-bench`가 scale을 `torch.Tensor`로 전달할 수 있음
- Triton 커널 인자로 `tl.constexpr`가 아닌 일반 scalar를 받으므로 Python float이어야 함
- `torch.Tensor`를 직접 전달하면 Triton이 type 해석에 실패할 수 있음

---

## 8. 흔한 실수와 디버깅

### 8.1 Variadic Arguments 금지

```python
# ❌ 빌더 검증 실패
def kernel(*args, **kwargs):
    ...

# ❌ *inputs 패턴도 실패
def kernel(*inputs, output, new_state):
    ...

# ✅ 모든 파라미터를 명시적으로 선언
def kernel(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state):
    ...
```

`flashinfer-bench`의 builder는 함수 시그니처를 inspect하여 파라미터 수를 검증한다. Variadic은 이 검증을 통과하지 못한다.

### 8.2 DPS 파라미터 수 불일치

```
Error: Destination-passing style callable: expected 11 parameters, but got 9
```

이 에러는 DPS가 활성화된 상태에서 output 파라미터를 시그니처에 포함하지 않았을 때 발생:

```python
# ❌ DPS인데 output 파라미터 없음 (9개)
def kernel(q, k, v, state, A_log, a, dt_bias, b, scale):
    return output, new_state

# ✅ DPS: output 파라미터 포함 (11개)
def kernel(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state):
    ...
```

### 8.3 k-last Layout에서의 Broadcasting 방향 실수

K-last layout `[BV, K]`에서 broadcasting 방향을 헷갈리기 쉽다:

```python
# State shape: b_h[BV, K]
# k shape: b_k[K]
# v shape: b_v[BV]

# ✅ 맞는 broadcasting
b_v -= tl.sum(b_h * b_k[None, :], axis=1)   # k@state: K 방향 reduce → [BV]
b_h += b_v[:, None] * b_k[None, :]           # outer(v, k): [BV,1] × [1,K] → [BV,K]
b_o = tl.sum(b_h * b_q[None, :], axis=1)     # q@state: K 방향 reduce → [BV]

# ❌ k-first layout과 혼동 (axis, broadcasting 방향 반대)
b_v -= tl.sum(b_h * b_k[:, None], axis=0)    # 이건 k-first [K, BV]에서의 패턴
```

**디버깅 팁:** axis 방향이 잘못되면 correctness는 실패하지만 커널은 crash하지 않는다. `abs_err`가 비정상적으로 크면 axis 방향을 의심하라.

### 8.4 Mask 누락

V 차원이 BV로 균등하게 나누어지지 않을 때를 대비한 mask:

```python
mask_v = o_v < V   # V=128, BV=128이면 항상 True이지만, BV가 V를 나누지 못하면 필수

b_h = tl.load(p_state, mask=mask_v[:, None], other=0.0)  # 2D mask: [BV, K]
b_v = tl.load(v_ptr + ..., mask=mask_v, other=0.0)        # 1D mask: [BV]
tl.store(output_ptr + ..., b_o, mask=mask_v)               # store에도 mask
```

**노하우:**
- `V % BV == 0`인 config에서는 mask가 항상 True → 컴파일러가 자동 제거
- 하지만 autotune config 중 `V % BV != 0`인 경우가 있을 수 있으므로 항상 mask를 포함
- `other=0.0`은 out-of-bounds 로드 시 0을 반환 — state의 빈 영역을 0으로 초기화

---

## 9. 추가 최적화 후보 (일부 적용)

### 9.1 `tl.make_block_ptr` (TMA Load) — ✅ 적용됨 (O16)

Hopper/Blackwell GPU에서 Tensor Memory Accelerator(TMA)를 활용한 block 단위 로드:

```python
# Before (pointer arithmetic + mask) — 제거됨
# p_state = state_ptr + ... + o_v[:, None] * K + o_k[None, :]
# b_h = tl.load(p_state, mask=mask_v[:, None], other=0.0)
# After — 현재 적용된 TMA 방식 (block pointer)
p_h = tl.make_block_ptr(
    state_ptr + (i_b * HV + i_hv) * V * K,
    shape=(V, K),
    strides=(K, 1),
    offsets=(i_v * BV, 0),
    block_shape=(BV, K),
    order=(1, 0),
)
b_h = tl.load(p_h, boundary_check=(0,))
```

**효과 (확인됨):**
- TMA 엔진이 주소 계산을 하드웨어에서 처리 — SM의 ALU 부담 감소
- Boundary check가 하드웨어에서 처리되어 mask 연산 불필요 (`o_v`, `mask_v` 변수 제거)
- B200에서 TMA가 async copy를 지원하여 latency hiding 개선 기대
- RTX 2070 SUPER에서는 일반 load로 fallback하여 성능 차이 없음 (Turing은 TMA 미지원)
- 코드 가독성 향상: pointer arithmetic + mask 패턴 → block pointer + boundary_check 패턴

**적용 범위:** state load [BV,K], state store [BV,K], v load [BV], output store [BV] — 총 4개 접근점

### 9.2 PTX Inline Softplus

fla-org에서 사용하는 PTX 레벨 softplus:

```python
# ex2.approx.ftz.f32 + lg2.approx.ftz.f32 사용
# exp(x) = 2^(x * log2(e))
# log(x) = log2(x) * ln(2)
```

**기대 효과:** PTX special function unit 직접 사용으로 1-2 instruction 절약

**미적용 이유:** Gate 계산은 head당 scalar 1회뿐 (총 8번). 전체 실행 시간에서 차지하는 비중이 무시할 만큼 작다.

### 9.3 `tl.math.exp2` / `tl.math.log2`

```python
# tl.exp(x) → tl.math.exp2(x * 1.4426950408889634)   # log2(e) = 1.4427...
# tl.log(x) → tl.math.log2(x) * 0.6931471805599453   # ln(2) = 0.6931...
```

**기대 효과:** `exp2`와 `log2`는 NVIDIA GPU에서 1 PTX instruction (`ex2.approx.f32`, `lg2.approx.f32`)

**미적용 이유:** 동일하게 scalar gate 계산에서만 사용되어 전체 성능 영향 미미.

### 9.4 Batch Size > 1 최적화

현재 모든 워크로드가 `batch_size=1`이므로 batch 차원 최적화를 하지 않았다. Batch가 커지면:

- Grid의 두 번째 차원 `B * HV`가 커져 SM 활용도가 자연스럽게 증가
- State read/write가 병목이 아닌 compute-bound로 전환될 수 있음
- Persistent kernel 패턴 (여러 batch를 하나의 block에서 순차 처리) 고려 가능

---

## 10. 벤치마크 결과 분석

### 10.1 RTX 2070 SUPER 결과 (로컬 테스트)

```
20/20 workloads PASSED
Latency: 0.058 ~ 0.070 ms (avg ~0.065 ms)
Speedup: 30.78x ~ 37.56x over reference
Max abs_err: 1.56e-02
```

### 10.2 성능 해석

**Speedup이 30x+인 이유:**

Reference 구현은 **순수 Python + PyTorch eager mode**:
- 각 연산마다 별도의 CUDA 커널 launch (overhead ~5µs × 연산 수)
- 중간 결과가 매번 global memory에 기록
- 최소 5개의 kernel launch: decay, matmul, subtraction, outer, output

우리 커널은 **5개 연산을 1개의 커널로 fuse**:
- 1회의 kernel launch overhead
- 중간 결과가 register에만 존재 (global memory 접근 최소화)
- State read(1회) + State write(1회) + Output write(1회)만 HBM 접근

**30x speedup의 의미:**
- 이것은 "순수 최적화"가 아니라 "fused kernel vs eager mode"의 차이
- 경쟁 상대도 비슷한 fused kernel을 구현할 것이므로, 실제 경쟁력은 **절대 latency**로 판단해야 함
- B200에서의 절대 latency가 최종 순위를 결정

### 10.3 오차 분석

| 워크로드 | abs_err | rel_err |
|----------|---------|---------|
| 최소 | 1.91e-05 | 1.07e-02 |
| 최대 | 1.56e-02 | 5.35e-01 |

**rel_err가 큰 이유:**
- 일부 output 값이 0에 매우 가까운 경우 `rel_err = |pred - ref| / |ref|`가 폭발
- **abs_err 기준으로는 모두 통과** — `flashinfer-bench`의 correctness 기준을 만족
- bf16 → f32 → bf16 변환 과정에서의 rounding 차이도 기여

### 10.4 B200에서의 기대 성능

| 항목 | RTX 2070 SUPER | B200 |
|------|----------------|------|
| HBM Bandwidth | 448 GB/s | ~8 TB/s |
| SM 수 | 40 | 192 |
| 커널 launch overhead | ~5µs | ~5µs (유사) |

- State 1MB 읽기/쓰기: RTX 2070 SUPER에서 ~2.2µs, B200에서 ~0.125µs
- 커널 launch overhead가 지배적이므로 **절대 latency 차이는 크지 않을 수 있음**
- BV 최적값이 달라질 수 있으므로 B200에서 autotune 재실행 필요

---

## 부록: 핵심 교훈 요약

| 번호 | 교훈 | 적용 |
|------|------|------|
| 1 | Reference 구현을 수학적으로 단순화하라 | outer product 2회 → 1회 |
| 2 | Scalar 연산은 vector 연산 전에 처리하라 | q pre-scaling |
| 3 | Gate 계산을 커널에 fuse하라 | 별도 커널 launch 제거 |
| 4 | K 전체를 하나의 block으로 하라 (K ≤ 256) | Reduction synchronization 제거 |
| 5 | In-place 연산으로 레지스터 압력을 줄여라 | 중간 변수 제거 |
| 6 | constexpr로 컴파일 타임 최적화를 활성화하라 | Dead code elimination, 상수 folding |
| 7 | 수치 안정성은 타협하지 마라 | Stable softplus, f32 state |
| 8 | GVA는 integer division으로 처리하라 | Memory 복사 제거 |
| 9 | Autotuning에 맡길 것과 고정할 것을 구분하라 | BV는 autotune, K는 고정 |
| 10 | Speedup 수치보다 절대 latency를 봐라 | 경쟁은 fused kernel끼리의 비교 |
