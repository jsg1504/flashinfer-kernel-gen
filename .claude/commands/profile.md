Profile the kernel to identify performance bottlenecks.

## Steps

1. **Pack solution** — Run `python scripts/pack_solution.py`.

2. **Choose profiling method** — Based on available tools:

   **Option A: NCU Profiling (if compute-sanitizer available)**
   ```python
   from flashinfer_bench.agents import flashinfer_bench_run_ncu
   from flashinfer_bench import Solution, TraceSet
   import json

   solution = Solution.from_json(json.load(open("solution.json")))
   # Load first workload
   traceset = TraceSet.from_file("mlsys26-contest/workloads/gdn/gdn_decode_qk4_v8_d128_k_last.jsonl")
   workload = traceset.workloads[0]

   output = flashinfer_bench_run_ncu(solution, workload, set="detailed", page="details", timeout=120)
   print(output)
   ```

   **Option B: Manual analysis (no NCU)**
   Analyze the kernel code directly for:
   - Memory bandwidth utilization
   - Register pressure
   - Occupancy estimation
   - Compute vs memory bound classification

3. **Analyze results** — Report in this format:

   ### Bottleneck Analysis

   | Metric | Value | Optimal | Utilization |
   |--------|-------|---------|-------------|
   | HBM Bandwidth | X GB/s | Y GB/s | Z% |
   | Compute (FLOPS) | X GFLOPS | Y GFLOPS | Z% |
   | Register/thread | N | 255 max | Z% |
   | Occupancy | N warps/SM | M max | Z% |

   **Classification**: [Memory-bound / Compute-bound / Launch-overhead-bound]

   **Bottleneck**: [Specific bottleneck description]

4. **Recommend optimization** — Based on the bottleneck:
   - Memory-bound → TMA loads, data layout, prefetching
   - Compute-bound → algorithmic simplification, PTX intrinsics
   - Launch-overhead-bound → persistent kernel, larger blocks, kernel fusion

5. **Update optimize.md** — Add any new optimization candidates (C7, C8, ...) to the "미적용 최적화 후보" table with expected impact and feasibility notes.
