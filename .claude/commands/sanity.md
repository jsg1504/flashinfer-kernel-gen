Run CUDA sanitizer checks on the kernel to detect memory and concurrency bugs.

## Steps

1. **Pack solution** — Run `python scripts/pack_solution.py`.

2. **Run sanitizers** — Execute via the flashinfer_bench API:

   ```python
   from flashinfer_bench.agents import flashinfer_bench_run_sanitizer
   from flashinfer_bench import Solution, TraceSet
   import json

   solution = Solution.from_json(json.load(open("solution.json")))
   traceset = TraceSet.from_file("mlsys26-contest/workloads/gdn/gdn_decode_qk4_v8_d128_k_last.jsonl")
   workload = traceset.workloads[0]

   output = flashinfer_bench_run_sanitizer(
       solution=solution,
       workload=workload,
       sanitizer_types=["memcheck", "racecheck", "synccheck", "initcheck"],
       timeout=300,
   )
   print(output)
   ```

   If the API is not available, try the command-line equivalent:
   ```bash
   compute-sanitizer --tool memcheck python scripts/run_local.py
   ```

3. **Report results** — For each sanitizer:

   | Sanitizer | Status | Issues |
   |-----------|--------|--------|
   | memcheck | ✅/❌ | N issues |
   | racecheck | ✅/❌ | N issues |
   | synccheck | ✅/❌ | N issues |
   | initcheck | ✅/❌ | N issues |

4. **Diagnose issues** — If any sanitizer reports problems:
   - Identify the specific memory access or race condition
   - Map to the kernel code location
   - Suggest a fix

5. **Fix and re-verify** — After fixes, re-run sanitizers and benchmark to confirm:
   - All sanitizers pass
   - Correctness maintained (20/20 workloads)
   - No performance regression
