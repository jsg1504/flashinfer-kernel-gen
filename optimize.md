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

---

## 미적용 최적화 후보

| # | 최적화 | 분류 | 기대 효과 | 미적용 사유 |
|---|--------|------|----------|------------|
| C1 | `tl.make_block_ptr` (TMA load) | Memory | HW 주소 계산으로 SM ALU 절감, async copy | 현재 성능 충분. B200에서 효과 검증 필요. |
| C2 | PTX inline softplus | Compute | `ex2.approx` + `lg2.approx`로 1-2 instruction 절약 | Scalar 연산 1회뿐, 전체 성능 영향 무시 가능. |
| C3 | `tl.math.exp2` / `tl.math.log2` | Compute | 1 PTX instruction으로 exp/log 처리 | C2와 동일 사유. |
| C4 | Persistent kernel | Parallelism | 큰 batch에서 SM 재활용 | 모든 워크로드가 batch_size=1. |
| C5 | K-tile 분할 (64+64) | Tiling | 레지스터 압력 감소 | K=128이 thread당 255 레지스터 한도 이내. 분할 시 sync 발생. |
| C6 | Warp specialization | Parallelism | Load/compute 분리 | T=1 단순 구조에서 overhead가 이득보다 큼. |
