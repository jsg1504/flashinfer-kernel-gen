Run a full kernel optimization iteration: analyze → plan → implement → verify → track.

## Pre-requisites

Load the `kernel-opt` skill before starting: `skill(name="kernel-opt")`

## Steps

1. **Load context** — Read these files in parallel:
   - `solution/triton/kernel.py` — current kernel implementation
   - `optimize.md` — previous optimization attempts and results
   - `know-how.md` — documented optimization techniques
   - `mlsys26-contest/definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json` — kernel spec

2. **Baseline benchmark** — Run `/project:bench` to establish current performance numbers.

3. **Identify bottleneck** — Analyze the kernel for the highest-impact optimization opportunity:
   - Check the "미적용 최적화 후보" section of `optimize.md` for candidates (C1–C6+)
   - Consider memory bandwidth utilization, register pressure, occupancy, launch overhead
   - If unsure, consult Oracle for architectural guidance

4. **Plan the change** — Before editing, state:
   - Which optimization (Ox or Cx from optimize.md) will be applied
   - Expected impact (latency reduction, bandwidth improvement, etc.)
   - Risk assessment (correctness risk, register pressure, etc.)
   - Ask the user for confirmation before proceeding

5. **Implement** — Edit `solution/triton/kernel.py`:
   - Make the minimal change for the optimization
   - Do NOT refactor unrelated code
   - Preserve all existing optimizations (O1–O15)
   - Keep comments up to date

6. **Verify correctness** — Run `/project:bench`:
   - All 20 workloads must PASS
   - abs_err should not increase significantly
   - If ANY workload fails, revert immediately and diagnose

7. **Compare performance** — Against the baseline from step 2:
   - Latency improved → proceed to tracking
   - Latency unchanged → evaluate if the optimization is worth the complexity
   - Latency regressed → revert and document why in optimize.md

8. **Track results** — Run `/project:track-opt` to record this attempt in `optimize.md`.

9. **Update docs** — If the optimization is novel (not already in know-how.md):
   - Add the technique to `know-how.md`
   - Update `optimize.md` optimization list if new Ox was added

## Revert Protocol

If correctness or performance regresses:
```bash
git checkout -- solution/triton/kernel.py
```
Then document the failed attempt in `optimize.md` with the reason.

## Important

- ONE optimization per iteration. Do not batch multiple changes.
- Always benchmark BEFORE and AFTER.
- Never suppress errors to make benchmarks pass.
