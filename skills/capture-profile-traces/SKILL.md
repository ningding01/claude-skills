---
name: capture-profile-traces
description: Capture GPU profiler traces from a running SGLang or vLLM inference server — three forms (combined prefill+decode, prefill by_stage, decode by_stage) at a chosen input length, with exact token-count control. Use when someone wants to profile a model (any model) on SGLang or vLLM to later break time down per module (pairs with the trace-module-xlsx skill). Works on ROCm/CUDA.
---

# Capture profiler traces (SGLang / vLLM)

Drive a running server's `/start_profile` + `/stop_profile` to produce, at a given
input length:
1. **combined** — one request, prefill + N decode steps, profiled together.
2. **prefill by_stage** — prefill on its own.
3. **decode by_stage** — decode on its own.

SGLang does #2/#3 exactly via `profile_by_stage`. vLLM has no stage tagging, so #2/#3
are request-shaped approximations (see `references/framework-notes.md`); on vLLM the
robust path is **combined + slice** in the analysis skill.

## Use this skill when

- You need traces to analyze where time goes (hand off to **trace-module-xlsx**).
- Comparing input lengths (4k vs 100k), GPUs (MI355 vs MI300X), or frameworks
  (SGLang vs vLLM) for the same model.

## Steps

1. **Launch the server with the profiler dir set.** Use `scripts/launch_server.sh`
   (sets `SGLANG_TORCH_PROFILER_DIR` / `VLLM_TORCH_PROFILER_DIR`, stack/shapes off):

   ```bash
   FRAMEWORK=sglang MODEL=/path/to/model TP=8 TRACE_DIR=/abs/traces PORT=8080 \
     EXTRA="--quantization mxfp8 --chunked-prefill-size 8192 --trust-remote-code \
            --reasoning-parser <p> --tool-call-parser <p>" \
     nohup bash scripts/launch_server.sh > server.log 2>&1 &
   ```
   (vLLM: `FRAMEWORK=vllm`, EXTRA holds vLLM flags; `--enable-prefix-caching` is added
   automatically so the decode capture can skip prefill.) Wait until the log says ready
   (SGLang `/health` returns 503 while warming, 200 when up). Loading + CUDA-graph
   capture for a big TP model can take several minutes.

2. **Capture.** One call does all three forms at one input length:

   ```bash
   python scripts/capture_traces.py --framework sglang --port 8080 \
     --out-dir /abs/traces --prefix m3_4k --input-len 4000 \
     --decode-steps 8 --combined-decode 32
   ```
   Repeat with `--input-len 100000 --prefix m3_100k` for another length.
   `--out-dir` MUST equal the server's profiler dir. The script waits for `/health`,
   sends exactly `--input-len` tokens (raw token ids), and prints the output file globs.

3. **Hand off to trace-module-xlsx** with the printed paths (prefer the by_stage files,
   fall back to combined).

## Key options

- `--input-len` exact prompt length (raw token ids, tokenizer-independent).
- `--decode-steps` (default 8): decode steps for the by_stage/decode capture — **keep
  small**; decode is CUDA-graphed so compute is recorded ~once and only sampling repeats,
  so few steps = clean ~1-step decode.
- `--combined-decode` (default 32): decode steps inside the combined capture.
- `--chunk` (default 8192): chunked-prefill size, used to size `num_steps` so the prefill
  EXTRA stage captures all chunks.
- `--id-mod` (default 20000): lower it only if the model vocab is < ~20100.
- `--model` (vLLM): served model name; auto-detected from `/v1/models` if omitted.

## Outputs

- **SGLang** (in `--out-dir`):
  - `<prefix>_combined-*-TP-*.trace.json.gz`
  - `<prefix>_stage-*-TP-*-EXTEND.trace.json.gz`  (prefill, by_stage)
  - `<prefix>_stage-*-TP-*.trace.json.gz`         (decode, by_stage)
- **vLLM** (sub-dirs under `--out-dir`): `<prefix>_combined/`, `<prefix>_prefill/`,
  `<prefix>_decode/` (prefill/decode are approximations).

## Caveats

- **Flush first / fresh prompt**: the script flushes the cache before the by_stage run so
  the prefill stage records the FULL prefill (not a 1-token cached extend).
- **Decode latency is not the trace total**: for absolute ms/step, time it
  `(t[N+1 tok]-t[1 tok])/N`; the trace is for composition.
- **Stop the server** when done to free GPUs.
- Deps: stdlib only. Server must expose the profile endpoints (set the profiler dir env
  at launch — `launch_server.sh` does this).
