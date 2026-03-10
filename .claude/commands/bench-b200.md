Pack the kernel solution and run the cloud benchmark on a real NVIDIA B200 GPU via Modal. Report results concisely.

## Prerequisites

- Modal CLI authenticated (`modal setup`)
- Volume `flashinfer-trace` created and populated with the dataset

## Steps

1. **Pack solution** — Run `python scripts/pack_solution.py` to generate `solution.json`.

2. **Run B200 benchmark** — Execute the Modal cloud benchmark:
   ```bash
   modal run scripts/run_modal.py
   ```
   This deploys to an NVIDIA B200 GPU on Modal. Expect 1-3 minutes for cold start + execution.

3. **Summarize results** — Present a table:

   | Metric | Value |
   |--------|-------|
   | Passed | N/20 |
   | Avg latency | X.XXX ms |
   | Speedup range | Nx ~ Nx |
   | Max abs_err | X.XXe-XX |
   | Max rel_err | X.XXe-XX |

4. **Compare with local** — If local benchmark results were recently run (via `/bench`), compare:
   - Latency difference (B200 vs local GPU)
   - Correctness consistency (same pass/fail pattern?)
   - Speedup factor changes

5. **Compare with previous B200 run** — If prior B200 results exist, note improvements or regressions.

6. **Suggest next action** — Based on results:
   - All passed + competitive speedup → "Ready for submission or further optimization"
   - Failures → "Correctness issue — debug locally first with `/bench`"
   - Perf below expectation → "Profile with `/profile` and check B200-specific tuning (block sizes, TMA)"
   - Modal error (cold start, OOM, timeout) → "Check Modal logs and resource config"
