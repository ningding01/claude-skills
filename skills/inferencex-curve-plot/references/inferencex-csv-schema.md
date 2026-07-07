# InferenceX CSV schema & the gotchas that bite curve plots

## benchmark_serving result JSON (this run)
`utils/bench_serving/benchmark_serving.py --save-result` writes one JSON per run with
(among others):
- `total_token_throughput`, `output_throughput`  (tokens/s, whole server — divide by TP for per-GPU)
- `mean/median/p99/std_ttft_ms`, `..._tpot_ms`, `..._itl_ms`, `..._e2el_ms`  (milliseconds)

Per-GPU throughput = `total_token_throughput / TP`.
Interactivity (tok/s/user) = `1000 / median_tpot_ms`.

## Official InferenceX CSV (baseline)
Standard schema (see `gen_csv.py::HEADER`). Rows carry `Precision`, `ISL`, `OSL`, `TP`,
`Concurrency`, `Spec Decoding`, plus `Throughput/GPU (tok/s)`, `Median Interactivity
(tok/s/user)`, `Median TTFT (ms)`, `Median TPOT (ms)`, `Median E2E Latency (ms)`, ...

### GOTCHA 1 — official time columns are in SECONDS despite the "(ms)" header
`Median TTFT (ms)` ≈ 0.21 for con1 8k1k → that's 0.21 **seconds**, not ms. Same for
`Median TPOT`, `Median E2E`. So `Throughput/GPU` and `Median Interactivity` are used
as-is, but any *time* column from the official CSV is already in seconds — do NOT divide
by 1000. (Filter rows by `Precision` and `ISL` before plotting.)

### GOTCHA 2 — don't compare `median_e2el` vs `Median E2E` column directly
median is not additive, and the two sides aggregate median-E2E differently: the official
`Median E2E` column is even lower than its own `TTFT + OSL×TPOT`. Plotting
`benchmark_serving.median_e2el` (this run) against the official `Median E2E` column
inflates the gap to ~10% even when throughput/TTFT/TPOT all match within ~1%.
**Fix:** recompute E2E for BOTH curves with the same formula
`E2E = TTFT + (OSL-1)·TPOT` using each side's median TTFT and median TPOT. Then the
E2E curves fit as tightly as the underlying TTFT/TPOT do. (`plot_curves.py` does this.)

### GOTCHA 3 — same-config comparison
Per-GPU throughput drops as TP grows (more comm, less compute/GPU). To compare against an
official TP4 curve, run at TP4 too. Also match the recipe: prefix caching, index cache,
`max-model-len`, and `--use-chat-template` all move the curve. If your curve sits above
official, first check these before claiming a win — the spec-decoding setting is usually
identical to official already.
