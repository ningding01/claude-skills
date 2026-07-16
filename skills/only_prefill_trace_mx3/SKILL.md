---
name: only_prefill_trace_mx3
description: Capture a pure-prefill torch-profiler trace for MiniMax-M3 (MXFP8) on AMD MI355 under the ATOM engine. Launches an ATOM OpenAI server with the profiler enabled (no EAGLE/speculative decode), drives it with a light-trace/agent-bench shared-prefix prefill workload (ISL ~75k = 67.5k shared + 7.5k unique, OSL=1, con20 saturated), and grabs a short steady-state profile window (/start_profile -> ~1.5s -> /stop_profile). Produces per-TP-rank *.pt.trace.json.gz traces for per-kernel/per-module analysis. Use when someone asks to "抓一版纯 prefill trace" / profile M3 prefill on MI355 at a given TP. Analysis/table-building is OUT OF SCOPE (see trace-module-xlsx).
---

# only_prefill_trace_mx3 — MiniMax-M3 pure-prefill trace capture (MI355 / ATOM)

Capture a torch-profiler trace whose GPU work is **prefill-dominated**, for MiniMax-M3
MXFP8 on 8×MI355 (gfx950) served by **ATOM**. The trace is meant for per-kernel /
per-module composition analysis and cross-config comparison (e.g. TP4 vs TP8).

This skill covers **capture only**. Turning traces into breakdown tables/xlsx is a
separate step — use the `trace-module-xlsx` skill.

## When to use

- "抓一版纯 prefill trace" for M3 on MI355, at some TP (4 or 8), no EAGLE.
- Need a prefill profile comparable to an existing one (same workload/chunk) at a
  different TP / KV dtype / config.

## What "pure prefill" means here (read this)

The workload is **generated-shared-prefix, OSL=1**: each request = a large shared
prefix (cache hit) + a unique tail (real prefill), generating exactly 1 token. At
con20 saturated the captured window is **prefill-dominated but not literally 0% decode**:
OSL=1 still runs one decode step per request, and with ~90% cache hit the unique tail
attends over the full KV via a paged path. A `paged_attention_decode_sliding_window`
kernel typically shows ~8% of the window. This is **consistent across configs** (same
workload), so it is fine for TP-vs-TP composition comparison. If you need literally
zero decode, that's a different capture (by_stage EXTEND-only) — not this skill.

## Prerequisites (verify first)

- `python3 -c "import atom"` works; `/app/ATOM` present; model at
  `/projects/models/MiniMax-M3-MXFP8`.
- light-trace / agent-bench checkout (default `/home/agslibadmin/niding/light-trace-benchmark`,
  run via `python3 -m agentbench.cli`). `TOKENIZERS_PARALLELISM=false`.
- GPUs free: `rocm-smi --showmeminfo vram` (idle ≈ 300 MB each). **Use a clean set of
  GPUs** — index cache + long context OOMs in AITER quick all-reduce if the GPUs are
  shared. For TP4 prefer GPUs 4-7.
- `gpu-memory-utilization 0.8` (index cache grows with `--max-model-len`; >0.8 can OOM).

## Procedure

All scripts are in `scripts/`. They are parameterized by env vars; defaults match the
proven runs. Work in a fresh output dir `$OUT` (holds `profile_out/`, `logs/`, `scripts/`).

1. **Launch the ATOM server with profiler on** (NO eagle):

   ```bash
   TP=4 GPUS=4,5,6,7 PORT=8000 OUT=/path/to/profile-tp4-run \
     bash scripts/launch_server.sh
   ```
   Key flags baked in: MXFP8 `ptpc_fp8`, `--kv_cache_dtype fp8`, `--block-size 128`,
   `--max-model-len 100000`, `--max-num-seqs 128`, **`--max-num-batched-tokens 8192`**
   (chunk prefill = 8192; keep IDENTICAL across configs you compare), M3 index cache
   (`use_index_cache`, `index_topk_freq:4`), `--torch-profiler-dir $OUT/profile_out`.
   Env: `ATOM_PROFILER_MORE=1` (record_shapes + with_stack — REQUIRED, else
   shape/BW/call-stack are empty), `ATOM_FORCE_ATTN_TRITON=1`,
   `AITER_QUICK_REDUCE_QUANTIZATION=INT4`. **No `--method eagle3 / --draft-model /
   --num-speculative-tokens`.**

2. **Wait for readiness** and confirm the engine kwargs match what you asked (esp.
   `tensor_parallel_size`, `max_num_batched_tokens: 8192`, `kv_cache_dtype`,
   `torch_profiler_dir`):
   ```bash
   OUT=/path/to/profile-tp4-run bash scripts/wait_ready.sh
   ```
   (Server load takes a few minutes; wait_ready polls `/health` up to 10 min.)

3. **Run the bench + capture the window** in one step (starts agent-bench, waits past
   ramp into steady state, fires `/start_profile` -> sleep 1.5s -> `/stop_profile`,
   then stops the bench):
   ```bash
   TP=4 PORT=8000 OUT=/path/to/profile-tp4-run \
     LT=/home/agslibadmin/niding/light-trace-benchmark \
     bash scripts/capture.sh
   ```
   The workload `scripts/workload_prefill_shared90_con20.yaml` = 75k ISL
   (67.5k shared + 7.5k unique), OSL=1, con20 saturated (offered qps >> capacity,
   backpressure via `max_inflight:20`), seed 42. Capture fires at ~cache-hit 90%.

4. **Validate** the trace before tearing down (so you can re-capture if bad):
   ```bash
   python3 scripts/validate_trace.py "$OUT"/profile_out/rank_0/*.pt.trace.json.gz
   ```
   Expect: thousands of GPU kernel events; top kernels = allreduce, `mfma_moe*`,
   `fmha_fwd*`, `_index_block_score_kernel`, `paged_attention_*`, `_fused_qkv_norm_rope`;
   a rough per-step estimate printed (fmha_fwd count / 3 dense layers = #steps).

5. **Stop the server** (SIGTERM then SIGKILL; verify GPU memory freed):
   ```bash
   bash scripts/cleanup.sh
   ```

## Output

`$OUT/profile_out/rank_{0..TP-1}/<model>_ts_<...>.pt.trace.json.gz` — one per TP rank,
kineto/perfetto. **rank_0 is representative.** View in https://ui.perfetto.dev or
`chrome://tracing`. Feed to `trace-module-xlsx` for a module breakdown.

## Comparing configs (e.g. TP4 vs TP8)

Re-run steps 1-5 with `TP=8 GPUS=0,1,2,3,4,5,6,7`. **Keep everything else identical**:
same `--max-num-batched-tokens 8192`, same workload, same KV dtype. Only then are
per-step / per-module numbers comparable. Note: absolute per-step time is best read
from the trace here (prefill is NOT under CUDA graph), but sanity-check with the bench's
own TTFT/throughput.

## Gotchas

- **Missing shapes in analysis** → you forgot `ATOM_PROFILER_MORE=1`.
- **OOM at init** (`quick_all_reduce.cuh`) → GPUs not clean, or util > 0.8, or
  max-model-len too big for index cache. Use clean GPUs, util 0.8.
- **Different chunk between configs** → not comparable. Pin `--max-num-batched-tokens`.
- **Trace too big** → shorten the capture window (sleep 1.0 instead of 1.5); do NOT cut
  concurrency (changes the prefill batch composition).
- **Server won't die via pkill -f** → kill by PID from the launch, then `pkill -9 -f
  atom.entrypoints.openai_server`; confirm with `rocm-smi`.
