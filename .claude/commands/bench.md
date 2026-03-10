Pack the kernel solution and run the local benchmark. Report results concisely.

## Steps

1. **Pack solution** — Run `python scripts/pack_solution.py` to generate `solution.json`.

2. **Run benchmark** — Run with the dataset path:
   ```bash
   FIB_DATASET_PATH=/home/jsg1504/workdir/flashinfer-kernel-gen/mlsys26-contest python scripts/run_local.py
   ```

3. **Summarize results** — Present a table:

   | Metric | Value |
   |--------|-------|
   | Passed | N/20 |
   | Avg latency | X.XXX ms |
   | Speedup range | Nx ~ Nx |
   | Max abs_err | X.XXe-XX |
   | Max rel_err | X.XXe-XX |

4. **Compare with previous** — If `optimize.md` exists, compare against the latest attempt's numbers and note improvements or regressions.

5. **Suggest next action** — Based on results:
   - All passed + good perf → "Ready for B200 benchmark or further optimization"
   - Failures → "Correctness issue — investigate error pattern"
   - Perf regression → "Consider reverting last change"
