---
name: inferencex-curve-plot
description: Turn InferenceX benchmark_serving result JSONs into throughput-latency curves — Throughput/GPU vs Interactivity and vs End-to-end Latency — optionally overlaying an official InferenceX CSV baseline for reproduction. Use when someone has a concurrency sweep (con 1..256) of benchmark_serving `--save-result` JSONs and wants the two standard InferenceX plots, an aggregated InferenceX-schema CSV, or a same-config comparison against the official dashboard data (e.g. MiniMax-M3 / DeepSeek 8k1k, 1k1k on MI355X/B200).
---

# InferenceX throughput–latency curves

From a concurrency sweep of `benchmark_serving.py` result JSONs → compute per-GPU
throughput / interactivity / E2E → (optional) aggregate to an InferenceX-schema CSV →
plot the two standard curves, optionally overlaid with the official InferenceX baseline.

## Use this skill when

- You ran an InferenceX-style concurrency sweep (`--max-concurrency N`, `--num-prompts N*10`,
  `--save-result`) at fixed ISL/OSL (e.g. 8k1k = ISL 8192 / OSL 1024) and want the plots.
- You want to reproduce/compare against the official InferenceX dashboard CSV (same model,
  ISL, precision) and show the curves overlaid.
- You need an aggregated CSV in the official schema to sit next to official data.

## Inputs

- **This run**: a directory of `benchmark_serving.py` result JSONs, one per concurrency;
  filename contains the concurrency (matched by a `--pattern` containing `{con}`),
  e.g. `m3_eagle3s3_isl8192_osl1024_con{con}.json`.
- **Baseline (optional)**: an official InferenceX CSV (standard schema). Filter with
  `--official-precision` (e.g. `fp8`) and `--official-isl` (e.g. `8192`).
- Know your run's **TP** (per-GPU normalization) and **OSL** (E2E formula).

## Steps

1. **Plot** (interactivity + e2e, points labelled by concurrency):
   ```bash
   python3 scripts/plot_curves.py \
     --results-dir /path/to/results --pattern 'm3_*_con{con}.json' \
     --cons 1,2,4,8,16,32,64,128,256 --tp 4 --osl 1024 \
     --out-dir /path/out --run-label 'this run (TP4)' \
     --title-suffix 'MiniMax-M3 · MI355X · ATOM · ISL=8k OSL=1k' \
     --official-csv official.csv --official-precision fp8 --official-isl 8192 \
     --official-label 'InferenceX official (fp8, TP4)'
   ```
   → `cmp_interactivity.png`, `cmp_e2e.png`.
2. **Aggregate CSV** (optional, InferenceX schema):
   ```bash
   python3 scripts/gen_csv.py --results-dir DIR --pattern 'm3_*_con{con}.json' \
     --cons 1,2,4,8,16,32,64,128,256 --tp 4 --isl 8192 --osl 1024 \
     --model MiniMax-M3 --hardware mi355x --framework atom --precision fp8 \
     --spec eagle3 --date 2026-07-07 --out repro.csv
   ```

## Metric definitions (get these right — see references/inferencex-csv-schema.md)

- **Throughput/GPU** = `total_token_throughput / TP`.
- **Interactivity** (tok/s/user) = `1000 / median_TPOT_ms`.
- **E2E** is recomputed for BOTH curves as **`TTFT + (OSL−1)·TPOT`** (each side's own
  median TTFT/TPOT). Do NOT plot benchmark_serving `median_e2el` against the official
  `Median E2E` column — their median aggregations differ and fake a ~10% gap.
- Official CSV **time columns (TTFT/TPOT/E2E) are in SECONDS** despite the "(ms)" header;
  throughput/interactivity columns are used as-is.

## Embedding plots in a shareable report

To send a single self-contained `.md` offline (images visible without shipping PNGs),
inline the figures as base64 data URIs:
```python
import base64
b = base64.b64encode(open("cmp_e2e.png","rb").read()).decode()
md = md.replace("](cmp_e2e.png)", f"](data:image/png;base64,{b})")
```
Local viewers (VS Code preview, Typora, browser) render data URIs; GitHub strips them.

## Same-config comparison (why a curve sits high)

Per-GPU throughput falls as TP rises — compare TP4 vs TP4. If your curve sits above the
official one, the cause is almost always **config, not a real win**: prefix caching,
`use_index_cache` hf-override, `max-model-len`, and client `--use-chat-template` all shift
the curve. The spec-decoding setting usually already matches official (e.g. eagle3
num-spec=3). Match the official recipe before claiming a difference.
