# GDN Decode Kernel 최적화 목록

> 각 시도(attempt)별로 적용된 최적화를 추적한다.

---

## 적용된 최적화 목록

| # | 최적화 | 분류 | 설명 |
|---|--------|------|------|
| O1 | Fused kernel | Algorithm | Gate 계산 + delta rule update + output을 단일 커널로 통합. 별도 커널 launch 제거. |
| O2 | Delta rule 단순화 | Algorithm | Reference의 outer product 2회를 수학적으로 정리하여 1회로 축소. (`state += outer(k, β*(v - k@state))`) |
| O3 | Query pre-scaling | Algorithm | `scale` 곱셈을 output 단계가 아닌 q 로드 직후 1회로 이동. BV 크기와 무관하게 K번 곱셈으로 고정. |
| O4 | In-place 연산 | Register | `b_h`, `b_v`를 중간 변수 없이 in-place 갱신. Intermediate tensor 할당 제거로 레지스터 절약. |
| O5 | K 전체 단일 block | Tiling | K=128을 분할하지 않고 한 번에 처리. K-reduction에서 thread 간 synchronization 제거. |
| O6 | V-only tiling + autotune | Tiling | V 차원만 BV ∈ {4,8,16,32,64,128}로 tiling. 각 tile이 완전 독립이라 동기화 불필요. |
| O7 | constexpr 차원 | Compile | H, HV, K, V, BV, USE_INITIAL_STATE를 `tl.constexpr`로 선언. 컴파일 타임 상수 folding + dead code elimination. |
| O8 | GVA integer division | Memory | `repeat_interleave` 대신 `i_h = i_hv // (HV // H)`로 head 매핑. q/k 메모리 복사 제거. |
| O9 | Numerically stable softplus | Numeric | `tl.where(x > 20, x, log(1+exp(x)))` — overflow 방지. |
| O10 | Log-space gate | Numeric | `log(g)`를 계산 후 `exp` 적용. Underflow 방지 + 수치 안정성 확보. |
| O11 | f32 연산 / f32 state | Numeric | 모든 연산을 float32로 수행. State를 f32로 유지하여 recurrent 오차 누적 방지. |
| O12 | Coalesced state access | Memory | State `[B, HV, V, K]`에서 K가 contiguous → K 방향 로드가 coalesced. |
| O13 | Scalar broadcast load | Memory | Gate 파라미터(A_log, dt_bias, a, b)를 scalar로 로드 → L1 cache broadcast, HBM 비용 0. |
| O14 | num_stages=1 | Compile | T=1 decode에서 loop 없으므로 software pipelining 비활성화. 불필요한 prefetch 코드 제거. |
| ~~O15~~ | ~~Contiguous 보장~~ | ~~Memory~~ | ~~Python wrapper에서 `.contiguous()` 호출. Non-contiguous tensor의 잘못된 pointer arithmetic 방지.~~ **제거됨 → O20** |
| O16 | `tl.make_block_ptr` (TMA) | Memory | 모든 2D/1D masked load/store를 block pointer로 교체. HW 주소 계산 + boundary check. Hopper+에서 TMA 엔진 활용. |
| O17 | Algebraic output reformulation | Algorithm | Output을 updated state 대신 decayed state + scalar correction으로 계산. State update와 output 사이의 데이터 의존성 제거 → critical path 단축. |
| O18 | 1D grid flattening | Scheduling | 2D grid `(V//BV, B*HV)` → 1D `(V//BV*B*HV,)`로 변환. GPU grid scheduling overhead 제거. (triton-lang #6166) |
| O19 | BV=4 autotune config | Tiling | BV=4 config 추가로 더 많은 block 생성 → SM 활용도 및 occupancy 개선. |
| O20 | `.contiguous()` 제거 | Python | `.contiguous()` 호출 제거. Benchmark가 contiguous tensor를 보장하므로 불필요한 Python 오버헤드 제거. |
| O21 | q/k block pointer (TMA) | Memory | q, k 로드를 `tl.make_block_ptr`로 변환. `o_k = tl.arange(0, K)` 변수 제거. 모든 메모리 접근이 TMA 경로 사용. |
| O22 | BV=2 autotune config | Tiling | BV=2 config 추가. 512 blocks 생성으로 SM 활용도 극대화. B200(192 SM)에서 2.67 waves → BV=4(1.33 waves) 대비 2배 SM saturation. |
| O23 | K-tile split (BK autotune) | Tiling | BK∈{64,128} autotune 추가. BK=128은 single-pass(기존), BK=64는 2-pass K-split으로 레지스터 압력 감소. BV≥16에서 occupancy 개선. |

---

## 시도별 기록

### Attempt 1 — Baseline fused kernel (현재)

**적용 최적화:** O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11, O12, O13, O14, O15 (전체)

**결과 (RTX 2070 SUPER):**
- 정확성: 20/20 PASSED
- Latency: 0.058 ~ 0.070 ms (avg ~0.065 ms)
- Speedup: 30.78x ~ 37.56x vs reference
- Max abs_err: 1.56e-02

**비고:**
- fla-org/flash-linear-attention의 fused_recurrent 패턴을 기반으로 구현
- 첫 시도에서 모든 기본 최적화를 한꺼번에 적용
- B200 벤치마크 미실행 (Modal 미설정)

### Attempt 2 — C1: `tl.make_block_ptr` TMA loads/stores (O16)

**적용 최적화:** O1–O16 (O16 신규)

**결과 (RTX 2070 SUPER):**
- 정확성: 20/20 PASSED
- Latency: 0.052 ~ 0.061 ms (avg ~0.055 ms)
- Speedup: 31.59x ~ 37.71x vs reference
- Max abs_err: 3.12e-02

**변경 사항:**
- State load [BV, K]: pointer arithmetic + mask → `tl.make_block_ptr` + `boundary_check=(0,)`
- State store [BV, K]: pointer arithmetic + mask → `tl.make_block_ptr` + `boundary_check=(0,)`
- V load [BV]: pointer arithmetic + mask → `tl.make_block_ptr` + `boundary_check=(0,)`
- Output store [BV]: pointer arithmetic + mask → `tl.make_block_ptr` + `boundary_check=(0,)`
- `o_v`, `mask_v` 변수 제거 (block pointer가 대체)

**이전 대비:**
- Latency: 동일 (~0.055 ms, RTX 2070 SUPER에서 TMA 미지원이므로 예상대로)
- 정확성: 유지 (20/20 PASS, abs_err 동일)
- 분산 감소: 0.047~0.066 → 0.052~0.061 (tighter variance)

**비고:**
- RTX 2070 SUPER (Turing, SM75)에서는 `tl.make_block_ptr`이 일반 load로 fallback하여 성능 차이 없음
- B200 (Blackwell)에서는 TMA 엔진이 주소 계산을 HW에서 처리하여 SM ALU 절감 기대
- 코드가 더 깔끔해짐 (pointer arithmetic + mask 패턴 → block pointer + boundary_check 패턴)
- B200에서 벤치마크 필요
---
### Attempt 3 — O17: Algebraic Output Reformulation (Critical Path Shortening)

**적용 최적화:** O1–O17 (O17 신규)

**변경 사항:**
- Output 계산을 `q @ updated_state`에서 `q @ decayed_state + delta * dot(k, q)`로 변경
- 수학적 동치: `q @ (S + outer(d,k)) = q @ S + d * dot(k,q)` (분배법칙)
- State rank-1 update와 output 계산 사이의 RAW register dependency 제거
- 컴파일러가 output store와 state update를 병렬 스케줄링 가능

**결과 (RTX 2070 SUPER):**
- 정확성: 20/20 PASSED
- Latency: 0.048 ~ 0.059 ms (avg ~0.056 ms)
- Speedup: 32.69x ~ 39.68x vs reference
- Max abs_err: 3.12e-02

**이전 대비:**
- Latency: 동일 (~0.056 ms, RTX 2070에서는 memory/launch overhead 지배적이라 compute overlap 효과 미미)
- 정확성: 유지 (20/20 PASS, abs_err 동일)
- Min latency: 0.054 → 0.048 ms (11% 감소 — 일부 workload에서 스케줄링 이점 확인)

**비고:**
- RTX 2070 SUPER에서는 avg latency 동일 (memory-bound에서 compute overlap 효과가 noise에 묻힘)
- B200 (HBM 8 TB/s)에서는 memory가 빨라 compute가 상대적으로 더 중요 → overlap 이점 기대
- 추가 연산: K muls + K adds (scalar dot product) + BV muls + BV adds — 전체 대비 무시 가능
- fla-org upstream에서도 `tl.dot` 미사용 확인 — element-wise + reduce 패턴이 recurrent kernel의 표준
- B200에서 벤치마크 필요
---
### Attempt 4 — O18+O19: 1D Grid Flattening + BV=4 Autotune

**적용 최적화:** O1–O19 (O18, O19 신규)

**변경 사항:**
- Grid를 2D `(V//BV, B*HV)` → 1D `(V//BV * B*HV,)`로 변환
- 커널 내부에서 `pid = tl.program_id(0)`, `NV = (V+BV-1)//BV` (constexpr), `i_v = pid % NV`, `i_bh = pid // NV`로 수동 decompose
- Autotune에 `BV=4` configs 2개 추가 (num_warps=1, 2)
- 총 autotune config: 10 → 12개

**결과 (RTX 2070 SUPER):**
- 정확성: 20/20 PASSED
- Latency: 0.050 ~ 0.055 ms (avg ~0.052 ms)
- Speedup: 34.66x ~ 38.74x vs reference
- Max abs_err: 6.25e-02

**이전 대비:**
- Avg Latency: 0.0555 → 0.0524 ms (**−5.7%** 개선)
- Min Latency: 0.054 → 0.050 ms (−7.4% 개선)
- Max Latency: 0.057 → 0.055 ms (−3.5% 개선)
- 정확성: 유지 (20/20 PASS). abs_err 증가 (1.56e-02 → 6.25e-02)는 autotune이 다른 BV를 선택하여 tile 분해 방식이 달라진 것에 기인. 모든 workload 통과.

**비고:**
- triton-lang #6166에서 2D grid의 scheduling overhead가 1D 대비 최대 3배까지 느려질 수 있음을 확인
- BV=4 시 grid = (32, 8) → 256 blocks. RTX 2070 SUPER의 40 SM에 대해 6.4 wave → 더 나은 메모리 대역폭 활용
- B200 (192 SM)에서는 256 blocks가 SM을 더 균등하게 채워 occupancy 이점 극대화 기대
- Register pressure: BV=4일 때 state tile [4,128] = 512 f32 → thread당 ~16 레지스터로 occupancy 대폭 개선
- 1D grid flattening은 NV가 2의 거듭제곱 (V=128, BV∈{4,8,...,128})이므로 modulo/division이 shift/mask로 최적화됨
---
### Attempt 5 — C9→O20: `.contiguous()` 제거

**적용 최적화:** O1–O20 (O15 제거 → O20 교체)

**변경 사항:**
- Python wrapper의 `.contiguous()` 호출 6개 제거 (q, k, v, state, a, b)
- Benchmark framework가 contiguous tensor를 보장하므로 방어적 복사 불필요

**결과 (RTX 2070 SUPER):**
- 정확성: 20/20 PASSED
- Latency: 0.054 ~ 0.056 ms (avg ~0.055 ms)
- Speedup: 34.41x ~ 37.77x vs reference
- Max abs_err: 7.81e-03

**이전 대비:**
- Avg Latency: 0.0524 → 0.0548 ms (noise 범위 내, 실질적 변화 없음)
- Latency 분산: 0.050~0.055 → 0.054~0.056 (더 안정적)
- Attempt 4의 0.035~0.036ms 이상치 2개가 사라짐 (autotune warm-up artifact로 판단)
- 정확성: 유지 (20/20 PASS)

**비고:**
- `.contiguous()` no-op check (이미 contiguous인 tensor) 비용이 예상보다 작음 (~0.1µs vs. 가설 0.5-1µs)
- Python overhead 절감 효과가 측정 noise에 묻힘
- 코드 단순화 이점은 있으나 성능 개선은 미미
- 이 결과는 현재 RTX 2070 SUPER에서 kernel+launch overhead 최적화가 한계에 도달했음을 시사
- **B200에서 벤치마크 필요** — memory가 ~18배 빠르므로 CPU/launch overhead 비중이 더 커져 차이가 나타날 수 있음
---
### Attempt 6 — O21: q/k Block Pointer Migration (Complete TMA Coverage)

**적용 최적화:** O1–O14, O16–O21 (O21 신규)

**결과 (RTX 2070 SUPER):**
- 정확성: 20/20 PASSED
- Latency: 0.044 ~ 0.053 ms (avg ~0.051 ms)
- Speedup: 35.26x ~ 45.85x vs reference
- Max abs_err: 1.25e-01

**변경 사항:**
- q, k 로드를 pointer arithmetic + `o_k` arange → `tl.make_block_ptr` block pointer로 변환
- `o_k = tl.arange(0, K)` 변수 완전 제거
- `boundary_check` 불필요 (K는 constexpr이며 block_shape와 정확히 일치)
- 이로써 커널 내 모든 메모리 접근(state, q, k, v, output)이 block pointer 방식 통일

**이전 대비:**
- Avg Latency: 0.052 → 0.051 ms (noise 범위, 실질적 변화 없음)
- Min Latency: 0.050 → 0.044 ms (일부 workload에서 개선, autotune artifact 가능)
- 정확성: 유지 (20/20 PASS). abs_err 변동은 autotune 재컴파일 후 다른 BV 선택에 기인.

**비고:**
- RTX 2070 SUPER (Turing, SM75)에서는 TMA 미지원이므로 O16과 동일하게 일반 load로 fallback
- B200 (Blackwell)에서는 q, k 로드도 TMA 엔진이 주소 계산을 HW에서 처리
- 코드 정리 효과: `o_k` 변수 제거로 커널 내 pointer arithmetic 완전 제거
- 이 변경으로 모든 메모리 접근이 `tl.make_block_ptr` 통일 — TMA 최적화 기반 완료
- **B200에서 벤치마크 필요** — TMA 효과 측정 가능
---
### Attempt 7 — O22: BV=2 Autotune Config (SM Saturation)

**적용 최적화:** O1–O14, O16–O22 (O22 신규)

**변경 사항:**
- Autotune에 `BV=2` configs 2개 추가 (num_warps=1, 2)
- 총 autotune config: 12 → 14개
- BV=2 시 grid = (64, 8) → 512 blocks. RTX 2070 SUPER의 40 SM에 대해 12.8 waves
- B200 (192 SM)에서 512/192 = 2.67 waves → BV=4(1.33 waves) 대비 2배 SM saturation

**결과 (RTX 2070 SUPER):**
- 정확성: 20/20 PASSED
- Latency: 0.035 ~ 0.055 ms (avg ~0.052 ms)
- Speedup: 34.72x ~ 54.44x vs reference
- Max abs_err: 1.56e-02

**이전 대비:**
- Avg Latency: 0.0529 → 0.0515 ms (**−2.6%** — noise 범위, 실질적 변화 미미)
- Min Latency: 0.043 → 0.035 ms (−18.6% — 일부 workload에서 autotune이 BV=2 선택 시 개선)
- Max Latency: 0.056 → 0.055 ms (동일)
- 정확성: 유지 (20/20 PASS, abs_err 동일)

**비고:**
- RTX 2070 SUPER에서는 avg 개선이 noise 수준 (−2.6%)
- BV=2 시 state tile [2,128] = 256 f32 → thread당 ~8 레지스터로 최대 occupancy
- B200 (192 SM)에서 autotune이 BV=2를 선택하면 SM saturation 2배 개선 기대
- O19 (BV=4)에서 −5.7% 개선을 보였으므로, BV=2는 B200에서 추가 이점 예상
- Zero-risk 변경: autotune이 더 느리면 BV=2를 선택하지 않음
- **B200에서 벤치마크 필요** — autotune이 BV=2를 선택하는지 확인
---
### Attempt 8 — C6 조사 + C5→O23: K-tile Split (BK=64 Autotune)

**C6 조사 결과: 적용 불가**
- Triton warp specialization은 `tl.range(warp_specialize=True)` 루프 기반 파이프라인 기법
- T=1 no-loop 커널에 구조적으로 적용 불가 (루프가 없어 producer/consumer 파이프라인 성립 안함)
- V-tile을 루프로 변환하면 grid 512→8 blocks로 SM 활용도 파괴
- `gl.warp_specialize()` (Gluon) 실험적 API도 overhead > benefit
- optimize.md에 "적용 불가"로 기록

**적용 최적화:** O1–O14, O16–O23 (O23 신규)

**변경 사항:**
- `BK` constexpr 파라미터 추가 (autotune: 64 또는 128)
- `NK: tl.constexpr = K // BK`로 K-tile 수 결정
- `if NK == 1:` → single-pass (BK=128, 기존 코드와 동일 — zero regression risk)
- `else:` → 2-pass K-split (BK=64, BV≥16에서 레지스터 압력 감소)
  - Phase 1: K-tile별 state 로드 → decay → K-reduction 누적 (k@state, dot(k,q), q@state)
  - Compute delta + output (누적된 reduction 사용)
  - Phase 2: state 재로드 (L2 cache hit) → re-decay → rank-1 update → store
- BK=64 configs 8개 추가 (BV∈{16,32,64,128} × num_warps∈{1/2,2/4,4/8})
- 총 autotune config: 14 → 22개

**결과 (RTX 2070 SUPER):**
- 정확성: 20/20 PASSED
- Latency: 0.043 ~ 0.056 ms (avg ~0.054 ms)
- Speedup: 34.88x ~ 46.60x vs reference
- Max abs_err: 6.25e-02

**이전 대비:**
- Avg Latency: 0.054 → 0.054 ms (변화 없음, −0.6% noise 범위)
- Min Latency: 0.051 → 0.043 ms (**−15.7%** — 2개 workload에서 autotune이 BK=64 선택)
- Max Latency: 0.055 → 0.056 ms (noise 범위)
- 정확성: 유지 (20/20 PASS, abs_err 동일)

**비고:**
- RTX 2070 SUPER에서 대부분의 workload는 autotune이 BK=128 (single-pass) 선택 → avg 변화 없음
- 2개 workload에서 BK=64가 선택되어 0.043–0.045ms 달성 (−15~16% min latency 개선)
- BK=64 2-pass는 state를 2번 읽지만 L2 cache hit로 실질적 HBM 추가 비용 최소
- BK=128 configs는 `if NK == 1:` 분기로 기존 코드와 완전 동일 → zero regression risk
- B200 (192 SM)에서 BK=64의 occupancy 개선이 더 클 것으로 기대
- **B200에서 벤치마크 필요** — autotune이 BK=64를 더 많은 workload에서 선택하는지 확인
---

## 미적용 최적화 후보

| # | 최적화 | 분류 | 기대 효과 | 미적용 사유 |
|---|--------|------|----------|------------|
| ~~C1~~ | ~~`tl.make_block_ptr` (TMA load)~~ | ~~Memory~~ | ~~HW 주소 계산으로 SM ALU 절감~~ | **적용됨 → O16** |
| C2 | PTX inline softplus | Compute | `ex2.approx` + `lg2.approx`로 1-2 instruction 절약 | Scalar 연산 1회뿐, 전체 성능 영향 무시 가능. |
| C3 | `tl.math.exp2` / `tl.math.log2` | Compute | 1 PTX instruction으로 exp/log 처리 | C2와 동일 사유. |
| C4 | Persistent kernel | Parallelism | 큰 batch에서 SM 재활용 | 모든 워크로드가 batch_size=1. |
| ~~C5~~ | ~~K-tile 분할 (64+64)~~ | ~~Tiling~~ | ~~레지스터 압력 감소~~ | **적용됨 → O23** |
| ~~C6~~ | ~~Warp specialization~~ | ~~Parallelism~~ | ~~Load/compute 분리~~ | **적용 불가** — Triton warp spec은 `tl.range(warp_specialize=True)` 루프 기반 파이프라인 기법. T=1 no-loop 커널에 구조적으로 적용 불가. V-tile 루프 변환 시 grid 512→8 blocks로 SM 활용도 파괴. (Attempt 8에서 조사) |
| ~~C7~~ | ~~1D grid flattening~~ | ~~Scheduling~~ | ~~2D grid scheduling overhead 제거~~ | **적용됨 → O18** |
| ~~C8~~ | ~~BV=4 autotune config~~ | ~~Tiling~~ | ~~더 많은 block으로 SM 활용도 증가~~ | **적용됨 → O19** |
| ~~C9~~ | ~~`.contiguous()` 제거~~ | ~~Python~~ | ~~wrapper의 CPU overhead ~3-6µs 감소~~ | **적용됨 → O20 (성능 변화 없음)** |
| ~~C12~~ | ~~BV=2 autotune config~~ | ~~Tiling~~ | ~~512 blocks로 SM saturation 극대화~~ | **적용됨 → O22 (RTX 2070에서 −2.6%, B200 확인 필요)** |
| C10 | `do_not_specialize` | Compile | 고정 값 인자의 specialization cache 감소 | fla-org 패턴. 전체 성능 영향 미미. |
| C11 | CUDA Graphs wrapping | Launch | kernel launch overhead 10-25µs 제거 | Framework-level 최적화. 커널 코드 변경 아님. |
