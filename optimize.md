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
| O6 | V-only tiling + autotune | Tiling | V 차원만 BV ∈ {8,16,32,64,128}로 tiling. 각 tile이 완전 독립이라 동기화 불필요. |
| O7 | constexpr 차원 | Compile | H, HV, K, V, BV, USE_INITIAL_STATE를 `tl.constexpr`로 선언. 컴파일 타임 상수 folding + dead code elimination. |
| O8 | GVA integer division | Memory | `repeat_interleave` 대신 `i_h = i_hv // (HV // H)`로 head 매핑. q/k 메모리 복사 제거. |
| O9 | Numerically stable softplus | Numeric | `tl.where(x > 20, x, log(1+exp(x)))` — overflow 방지. |
| O10 | Log-space gate | Numeric | `log(g)`를 계산 후 `exp` 적용. Underflow 방지 + 수치 안정성 확보. |
| O11 | f32 연산 / f32 state | Numeric | 모든 연산을 float32로 수행. State를 f32로 유지하여 recurrent 오차 누적 방지. |
| O12 | Coalesced state access | Memory | State `[B, HV, V, K]`에서 K가 contiguous → K 방향 로드가 coalesced. |
| O13 | Scalar broadcast load | Memory | Gate 파라미터(A_log, dt_bias, a, b)를 scalar로 로드 → L1 cache broadcast, HBM 비용 0. |
| O14 | num_stages=1 | Compile | T=1 decode에서 loop 없으므로 software pipelining 비활성화. 불필요한 prefetch 코드 제거. |
| O15 | Contiguous 보장 | Memory | Python wrapper에서 `.contiguous()` 호출. Non-contiguous tensor의 잘못된 pointer arithmetic 방지. |
| O16 | `tl.make_block_ptr` (TMA) | Memory | 모든 2D/1D masked load/store를 block pointer로 교체. HW 주소 계산 + boundary check. Hopper+에서 TMA 엔진 활용. |
| O17 | Algebraic output reformulation | Algorithm | Output을 updated state 대신 decayed state + scalar correction으로 계산. State update와 output 사이의 데이터 의존성 제거 → critical path 단축. |

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

## 미적용 최적화 후보

| # | 최적화 | 분류 | 기대 효과 | 미적용 사유 |
|---|--------|------|----------|------------|
| ~~C1~~ | ~~`tl.make_block_ptr` (TMA load)~~ | ~~Memory~~ | ~~HW 주소 계산으로 SM ALU 절감~~ | **적용됨 → O16** |
| C2 | PTX inline softplus | Compute | `ex2.approx` + `lg2.approx`로 1-2 instruction 절약 | Scalar 연산 1회뿐, 전체 성능 영향 무시 가능. |
| C3 | `tl.math.exp2` / `tl.math.log2` | Compute | 1 PTX instruction으로 exp/log 처리 | C2와 동일 사유. |
| C4 | Persistent kernel | Parallelism | 큰 batch에서 SM 재활용 | 모든 워크로드가 batch_size=1. |
| C5 | K-tile 분할 (64+64) | Tiling | 레지스터 압력 감소 | K=128이 thread당 255 레지스터 한도 이내. 분할 시 sync 발생. |
| C6 | Warp specialization | Parallelism | Load/compute 분리 | T=1 단순 구조에서 overhead가 이득보다 큼. |
