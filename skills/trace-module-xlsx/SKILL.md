---
name: trace-module-xlsx
description: Turn GPU profiler traces (kineto/perfetto .json/.json.gz, e.g. from SGLang/vLLM torch profiler on ROCm/CUDA) into an Excel module-breakdown table where kernels are grouped by the MODEL's own call structure, with per-module GPU-busy ms/%, and side-by-side comparison columns. Use when comparing the same model across GPUs (MI355 vs MI300X), across frameworks (SGLang vs vLLM), or both, or splitting prefill vs decode, or asking "where does the time go per module".
---

# Trace → module-breakdown xlsx

Classify every GPU kernel in a trace into **model-native modules** (not a fixed
scheme), sum GPU-busy time per module, and emit an xlsx with side-by-side
comparison columns + per-kernel detail. Mirrors the layout of a per-step module
table (module | ms | % | … | ratio) with subtotals.

## Use this skill when

- Someone hands you one or more profiler traces and wants a per-module breakdown.
- Comparing the **same model** across: different GPU, different framework, or both.
- Splitting **prefill vs decode** from a combined trace.
- The user references a module scheme like `dense_gemm (q/kv/o + shared expert)` and
  wants the table "according to the model's own kernels".

## Inputs

- Trace files: kineto/perfetto chrome traces, `.json` or `.json.gz`, one per TP rank
  (named `...TP-<n>...`). SGLang writes them to `SGLANG_TORCH_PROFILER_DIR` via
  `/start_profile` + `/stop_profile`. Point at the **TP-0** file; set `avg_ranks:true`
  to average across all ranks.
- Two kinds of trace per phase, and the skill **prefers by_stage**:
  - **profile_by_stage** files (preferred): one phase per file, exact boundary.
    Captured with `profile_by_stage:true` in `/start_profile` → writes
    `<prefix>-TP-<n>-EXTEND.trace.json.gz` (**prefill**) and
    `<prefix>-TP-<n>.trace.json.gz` (no suffix = **decode**).
  - **combined** trace (fallback): prefill+decode in one file; the skill slices it by
    phase using decode-marker kernels.
- Label each trace with `model`, and (for the header) `framework` + `gpu`.

## Spec fields (per dataset)

- `label`  — unique key (referenced by `sheets[].compare`).
- `display`— clean header name (lets two sheets both show "4k"/"100k" while labels stay unique).
- `phase`  — `prefill` | `decode` (needed for the combined-slice fallback).
- `stage_trace`    — glob to the by_stage file for this phase (**preferred**; used if it
  matches any file). prefill → the `*-EXTEND*` file; decode → the no-suffix file.
- `combined_trace` — glob to the combined trace (**fallback**; sliced by `phase`).
- `model`, `framework`, `gpu`, `avg_ranks`.
- Legacy form still works: `trace` + `region:"all|prefill|decode"`.

Source preference per dataset: `stage_trace` (if files exist) → else `combined_trace`
(sliced by `phase`) → else `trace`. The `--coverage` line prints which source was used,
e.g. `[minimax-m3/by_stage]` or `[minimax-m3/combined-slice(fallback)]`.

## How to run

1. Write a spec JSON (copy `examples/spec_example.json`). One `datasets` entry per
   (trace, region); list which datasets each comparison sheet shows.

2. **Always run coverage first** — especially for a model/framework not seen before:

   ```bash
   python scripts/build_xlsx.py myspec.json --coverage
   ```

   Check `unclassified %` per dataset. If any is > ~3%, the rule set doesn't fit that
   model/framework yet → extend `scripts/rulesets.py` (see `references/extending.md`),
   re-run coverage until small.

3. Generate the workbook:

   ```bash
   python scripts/build_xlsx.py myspec.json
   ```

   Produces one summary sheet per `sheets[]` entry (modules × datasets, ms/%/ratio,
   TOTAL row) **plus a matching `det <title>` sheet** (if `detail:true`) where the
   compared datasets are laid out **side by side** (one 5-col block each: module |
   kernel | ms | % | calls, with per-module subtotals) — e.g. `det prefill` and
   `det decode`, each with 4k on the left and 100k on the right.

Dependencies: `openpyxl` (`pip install openpyxl -q` if missing). Engine is stdlib.

## Module scheme is per model

Rule sets live in `scripts/rulesets.py`, keyed by model family. Shipped:
`minimax-m3`, `deepseek-v4`, `generic` (model-agnostic fallback). `resolve_ruleset()`
maps a free-form `model` label to one. Each rule set's module **names reflect that
model** (M3: lightning indexer / top-k select / sparse main attention / SwiGLU-OAI MoE;
DSV4: MLA + NSA compressor + MHC sparse core). Patterns intentionally cover multiple
frameworks and GPUs so cross-framework / cross-GPU rows align. Adding a model = a small
new block; see `references/extending.md`.

## Critical correctness notes

- Numbers are **GPU-busy ms** (sum of kernel durations) for the chosen region — not
  wall-clock. Good for *composition*; for absolute latency also measure by timing.
- **Prefill region** = whole prompt (may be several chunked steps; that's expected).
- **Decode region under CUDA graph** = the trace records the graph ~**once**, so the
  region total ≈ ONE step and matches measured ms/step. **Never divide a decode region
  by the generated-token count.** Sanity-check with `(t[N+1 tok]-t[1 tok])/N` timing.
- `region:"prefill"|"decode"` (combined-slice) needs the ruleset's `decode_markers`.
  **Markers must include the model's DENSE-layer decode attention**, not only the sparse
  decode kernels: dense layers run FIRST each step, so if only sparse markers are listed
  the boundary lands after them and decode loses its dense full-attention (it leaks into
  prefill). For minimax-m3 the markers already include `_fwd_grouped_kernel_stage1` /
  `_fwd_kernel_stage2`. **Prefer by_stage to sidestep this entirely.**
- **by_stage decode + step count**: a stage decode file aggregates ALL decode steps in
  the run. CUDA-graph *compute* is recorded ~once (1 step), but *eager* per-step kernels
  (sampling) appear ×N → the `elementwise/…` bucket inflates with N. For a clean per-step
  decode via by_stage, capture **few** decode steps (`max_new_tokens` ~2–4). The
  combined-slice decode is naturally ~1 step.
- Ratio columns are vs the **first** dataset in each sheet.

## Quick recipes

- same model, MI355 vs MI300X: two datasets (same region), one sheet, set `gpu` each.
- SGLang vs vLLM (same GPU): two datasets with different `framework` + traces; if vLLM
  kernel names are unclassified, extend the model's rule set with vLLM names.
- prefill+decode for one run: 4 datasets (2 regions × 2 configs), 2 sheets.
