# SGLang vs vLLM profiling — what differs

| | SGLang | vLLM |
|---|---|---|
| profiler dir env | `SGLANG_TORCH_PROFILER_DIR` | `VLLM_TORCH_PROFILER_DIR` |
| start / stop | `POST /start_profile` (JSON body) / `POST /stop_profile` | `POST /start_profile` (no body) / `POST /stop_profile` |
| exact token input | `POST /generate {input_ids:[...]}` | `POST /v1/completions {prompt:[token ids]}` |
| force decode length | `sampling_params.ignore_eos` + `max_new_tokens` | `ignore_eos` + `max_tokens` |
| **profile_by_stage** | **YES** (`profile_by_stage:true`, `num_steps`) → writes `*-EXTEND*` (prefill) and no-suffix (decode) | **NO** |
| cache flush | `POST /flush_cache` | `POST /reset_prefix_cache` |
| filename control | `profile_prefix` in start body | none (auto names) → capture moves files into a sub-dir |
| trace files | one per TP rank: `...TP-<n>...trace.json.gz` | per rank under the dir |

## Why the three captures

- **combined**: realistic single request (prefill + N decode) in one trace. Always works
  on both frameworks. The analysis skill can slice it into prefill/decode.
- **prefill by_stage** / **decode by_stage** (SGLang): one `profile_by_stage` capture
  tags each forward pass by ForwardMode and writes prefill (`-EXTEND`) and decode
  (no-suffix) separately — exact boundary, no marker guessing. Flush the cache first so
  `-EXTEND` records the FULL prefill (not a 1-token cached extend).

## vLLM has no stage tagging → approximations

- **prefill (approx)**: `max_tokens=1` → the trace is prefill + 1 decode step.
- **decode (approx)**: warm the prompt once (prefix cache), then profile a 2nd identical
  request → prefill is a cache hit (~1 token) so the window is mostly decode. Requires
  `--enable-prefix-caching`. Note: still includes a 1-token extend; the analysis skill's
  marker slice removes it. Honestly, for vLLM prefer **combined + slice**.

## Decode + CUDA graph (both frameworks)

Decode usually runs under a CUDA/HIP graph, so the trace records the graphed compute
~**once** regardless of how many tokens were generated; only EAGER per-step kernels
(sampling) repeat. So:
- Keep `--decode-steps` SMALL (2–8) for the by_stage/decode capture → minimal sampling
  inflation, clean ~1-step decode.
- For absolute per-step latency, also TIME it: `(t[N+1 tok]-t[1 tok])/N`, don't read it
  off the trace.

## Token ids

We send raw ids `(i % id_mod)+id_base` (default mod 20000, base 100) to hit an exact
length without depending on a tokenizer. If a model's vocab is < ~20100, lower `--id-mod`.

## Hand-off to the analysis skill

Feed the captured files to **trace-module-xlsx**: prefer the by_stage files
(`stage_trace`), fall back to the combined trace (`combined_trace` + `phase`). For vLLM,
point `combined_trace` at the `*_combined/` file (and/or the `*_prefill`/`*_decode`
sub-dirs as `stage_trace` if you trust the approximation).
