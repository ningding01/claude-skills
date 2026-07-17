---
name: m3-mxfp8-sglang-qc
description: Run MiniMax-M3-MXFP8 supplier quality inspection (Format Correctness + Basic Check) against a standalone SGLang server on AMD ROCm (MI355X/gfx950, TP4). Use when asked to QC / 质检 / verify an M3-MXFP8 SGLang deployment with the MiniMax-Provider-Verifier, launch the server for it, diagnose failures, or reproduce the known ROCM_QUICK_REDUCE_QUANTIZATION=INT4 garbled-output issue.
---

# MiniMax-M3-MXFP8 · SGLang 质量检测 Skill

End-to-end supplier QC for MiniMax-M3-MXFP8 on **standalone SGLang + AMD ROCm**, aligned to
`M3 External Provider Quality Inspection Manual`. Covers the two self-inspection items that are
deterministic and compute-cheap: **Format Correctness (text)** and **Inference Quality → Basic Check**.

Bundled scripts (in `scripts/`):
- `launch_official_tp4.sh` — **official upstream** (`sgl-project/sglang`) gfx950/MI355X launch (cookbook mi355x recipe + the `--enable-aiter-allreduce-fusion` bug workaround + thread caps). Env-overridable: `API_KEY=`, `CACHE_REPORT=1` add `--api-key`/`--enable-cache-report` for the config-fixable Format items. **Use this on upstream.** See §2.1.
- `launch_sglang_tp4.sh` — **ATOM fork** (`m3-atom-prefill-port`) launch (many `SGLANG_MINIMAX_M3_*` env). Use this only on that fork.
- `run_basic_check_resilient.sh` — crash-resilient Basic Check (auto-restart + `--incremental`)

## 0. Prerequisites
- SGLang installed **editable** (`pip show sglang` → "Editable project location"); branch supporting `minimax_m3_vl`.
- aiter present under `/sgl-workspace/aiter` (JIT-compiled kernels).
- `MiniMax-Provider-Verifier` checkout (has `verify.py`, `sample.jsonl` 102 samples, `m3_format_check/m3_text_tests.py` 150 cases).
- Deps: `pip install pytest pytest-xdist pytest-timeout httpx jsonschema loguru megfile tqdm`.
- GPUs free (check `rocm-smi --showmemuse`); pick idle cards.

## 1. Pre-flight (do these BEFORE launching — they are the top failure causes)
1. **Raise cgroup pids.max** (defaults to ~2048; a 384/768-core box explodes past it → `Resource temporarily unavailable (thread.cpp)` / `RuntimeError: can't start new thread` / `Rank N scheduler died (exit code -6)` at Gloo init):
   - cgroup **v2**: `echo 200000 > /sys/fs/cgroup/pids.max`
   - cgroup **v1**: real file is `/sys/fs/cgroup/pids/pids.max` (NOT `/sys/fs/cgroup/pids.max`, which is a dummy tmpfs file).
   - **If the cgroup is mounted read-only** (`.../pids type cgroup (ro,...)`), you CANNOT raise it — you MUST cap threads instead (this is the common container case): `export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 RAYON_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false` and launch with `SGLANG_SET_CPU_AFFINITY=1`. With TP4 this keeps `pids.current` ~600 (< 2048). The launch scripts already set these.
   For your own diagnostic `python3 -c`/`git` commands also `export OPENBLAS_NUM_THREADS=1` or they segfault the same way.
2. Confirm model snapshot path, free GPUs, and free port.
3. **Orphan cleanup between (re)launches**: a killed launcher leaves `sglang::scheduler` children that hold GPU/port. `pkill -9 -f 'sglang.launch_server'; pkill -9 -f 'sglang::'`, then verify `rocm-smi` shows the cards free. Ignore `<defunct>`/`Z` (zombie) procs — they hold nothing.

## 2. Launch the server
```bash
echo 200000 > /sys/fs/cgroup/pids.max
GPUS=4,5,6,7 PORT=8043 \
SNAP=/path/to/models--MiniMaxAI--MiniMax-M3-MXFP8/snapshots/<hash> \
bash scripts/launch_sglang_tp4.sh
```
Key choices already baked into the script (all env-overridable):
- **`QR_QUANT=none`** → does NOT set `ROCM_QUICK_REDUCE_QUANTIZATION`. **CRITICAL** — see §6.1.
- `--context-length 1048576` → covers the 512K/1M Format cases (`test_06_09` needs max_tokens=524288 accepted; `test_17_*` need 512K input).
- `--disable-cuda-graph` → avoids an aiter JIT `make` compile race across TP ranks during decode-graph capture (`lib.so: cannot open shared object file`). QC doesn't need cuda-graph perf.
- `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS` capped → belt-and-suspenders for the thread limit.
- `--tool-call-parser auto --reasoning-parser auto` → auto-detect `minimax-m3` from chat_template markers (`<mm:think>`, `]<]minimax[>[`). Verify the log shows `Auto-detected ... as 'minimax-m3'`.

Wait for health (M3 = 31 shards, ~7-10 min):
```bash
for i in $(seq 1 60); do [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8043/health)" = 200 ] && break; sleep 10; done
```

### 2.1 Which sglang build? Official upstream vs the ATOM fork — DIFFERENT launch
`scripts/launch_sglang_tp4.sh` is the **ATOM fork** (`m3-atom-prefill-port`) recipe (many `SGLANG_MINIMAX_M3_*` env). If you are on **official upstream** sglang (`sgl-project/sglang`, gfx950/MI355X native MXFP8), use the **cookbook mi355x recipe instead** — it is much smaller and the ATOM env vars don't apply:
```bash
SGLANG_USE_AITER=1 HIP_VISIBLE_DEVICES=4,5,6,7 python -m sglang.launch_server \
  --model-path <snap> --trust-remote-code --reasoning-parser auto --tool-call-parser auto \
  --tp-size 4 --quantization mxfp8 --dtype bfloat16 \
  --enable-aiter-allreduce-fusion \            # 🔴 REQUIRED workaround, see below
  --chunked-prefill-size 8192 --mem-fraction-static 0.80 --context-length 1048576 \
  --host 0.0.0.0 --port 8043
```
- 🔴 **Upstream AMD launch bug (as of 2026-07, PR #28715 / commit `0663ebc783`)**: the official recipe **crashes at arg-parse**:
  `ValueError: _minimax_m3_overrides: ['disable_custom_all_reduce'] not model-overridable`.
  Cause: on ROCm with `enable_aiter_allreduce_fusion=False` (default), the M3 override declares `disable_custom_all_reduce`, a field lacking `resolvable=True`. **Workaround (no code change): pass `--enable-aiter-allreduce-fusion`** (skips that declaration). Alternatively add `resolvable=True` to the field. The ATOM fork does NOT have this gate, so it needs no workaround.
- To check the arch: `rocm-smi --showproductname | grep gfx` (gfx950=MI355X native MXFP8; gfx942=MI300X → also add `--attention-backend aiter --moe-runner-backend triton --watchdog-timeout 3600 --skip-server-warmup`).
- `--dtype bfloat16` is part of the official gfx950 recipe (non-quantized activations).

## 3. Smoke test (confirm 3 fields parse)
Send a basic chat, a chat with `tools`, and check the response has: `content`, `reasoning_content`, and (for the tool req) `finish_reason=tool_calls` + parsed `tool_calls[].function.name/arguments`. If tool_calls don't parse, set `TOOL_PARSER=minimax-m3 REASONING_PARSER=minimax-m3` explicitly and relaunch.

## 4. Run the checks
Set the shared env first:
```bash
export M3_BASE_URL=http://127.0.0.1:8043 M3_API_KEY=dummy M3_MODEL=<served model/path>
```
**Basic Check** (verify.py, 102 samples) — use the resilient runner (the server WILL crash mid-run, see §6.2):
```bash
VERIFIER=/path/MiniMax-Provider-Verifier OUTDIR=/path/qc/raw \
LAUNCH="$PWD/scripts/launch_sglang_tp4.sh" M3_MODEL="$M3_MODEL" \
bash scripts/run_basic_check_resilient.sh
```
Metrics come from `basic_check_summary.json`: `success_rate`, `tool_calls_match_rate` (=(TP+TN)/100), `tool_calls_schema_validation_error_count`, `error_only_reasoning_rate`, `language_following_*`, `scenario_check_pass_rate`. verify.py force-sets M3 `max_tokens=40960` (baseline-aligned; your `--extra-body` cap is overridden — do not rely on it).

**Format Correctness (text)** (pytest, 150 cases):
```bash
cd $VERIFIER/m3_format_check
M3_RUN_LOG=/path/qc/raw/format_text_run.jsonl \
python3 -m pytest m3_text_tests.py -n 4 -v --junitxml=/path/qc/raw/format_text_junit.xml
```
Note `-n 8` is faster but higher crash risk; if the server dies near the end, in-flight tests get false ConnectionError failures — re-run just the failed node ids in isolation to separate real vs transient.

## 4a. 🔧 Launch-param ↔ test-item matrix (配错参数 = 假失败;先按此配全再判断)
Set these on the **server launch** BEFORE running Format Check, or the listed items fail for config reasons (not real gaps). Verified 2026-07 on MI355X, upstream 0.5.15 AND the ATOM fork — behavior identical.

| Launch param / env | Without it → these Format items FAIL | Why |
|---|---|---|
| `--api-key <key>` (client sends it via `M3_API_KEY`) | `20_05_no_authorization`, `20_07_invalid_api_key` | no key configured → server returns 200 for missing/bad auth instead of 401 |
| `--enable-cache-report` | `10_04_cached_tokens_presence` | response omits `usage.prompt_tokens_details.cached_tokens` |
| `--context-length 1048576` | `06_09` (max_tokens=524288), `17_*` (512K input) | request rejected for exceeding context |
| `--tool-call-parser auto` + `--reasoning-parser auto` | all `13_*`/`14_*` tool cases, reasoning-split cases | tool_calls / reasoning_content won't parse (must log `Auto-detected ... 'minimax-m3'`) |
| **NOT** setting `ROCM_QUICK_REDUCE_QUANTIZATION` (keep `QR_QUANT=none`) | long/complex gens: many `07_*`/`12_*`/`13_*` + Basic ToolCalls/Scenario | INT4 all-reduce garbles long output → runaway/token-soup = mass false fails (§6.1) |
| `--enable-aiter-allreduce-fusion` (upstream AMD only) | *server won't launch at all* | §2.1 arg-parse bug workaround |
| thread caps + `SGLANG_SET_CPU_AFFINITY=1` (pids.max readonly) | *server won't launch* (Rank died -6) | §1 |

So a **clean Format run** on upstream AMD = the §2.1 recipe **plus** `--api-key dummy --enable-cache-report`. With those, the config-fixable items pass; run `export M3_API_KEY=dummy` so the client authenticates.

### ⛔ NOT fixable by any launch param (genuine SGLang engine/parser/model gaps — don't chase them)
These fail on **both** upstream and the fork regardless of flags; needing code changes, not config:
| Item(s) | Gap |
|---|---|
| `11_01`/`11_02`/`11_04` role=root (×[ns,s], 6) | API schema rejects `role=root` (400) — role enum has no `root` (`protocol.py`) |
| `20_02` invalid_model, `20_03` temperature>upper-bound, `16_08` tool_call_id_mismatch, `16_09` partial_tool_call | engine returns 200 instead of 4xx (no model check; temperature has no upper bound in `sampling_params.py`; no tool_call_id-pairing validation). *(upstream PR #31419 in-flight for invalid_model→404.)* |
| `06_08` max_tokens_negative | 400 body lacks `trace_id` field (`ErrorResponse` schema) |
| `14_07` oneOf toplevel schema | `minimax_m3.py` parser doesn't resolve oneOf/anyOf; number not typed / array markers leak |
| `04_01` thinking_disabled, `13_08` tool_choice_values, `13_12` tool_name_mismatch | model behavior (sampling/instruction-following); `13_08` is flaky under temp=1.0 |

→ When these fail, **record as engine/parser/model gap and move on** — do NOT retune launch flags. All are platform-independent (no cuda/hip branch), so NVIDIA would fail them identically.

## 5. Classify failures (always do this — raw pass/fail is misleading)
For every failure decide which bucket:
- **skipped** = N/A (e.g. `TestResponseFormat` json_object — M3 doesn't support it; suite auto-skips).
- **crash/concurrency transient** → re-run in isolation; if it passes, it's not a real failure.
- **runaway/garbled (INT4)** → see §6.1; disappears once `ROCM_QUICK_REDUCE_QUANTIZATION` is dropped.
- **genuine gap** → still fails on the clean config. Sub-split: config-fixable (`--api-key`, `--enable-cache-report`) vs engine/protocol gap (needs code changes).

Extract failed node ids from junit, re-run with `-n 4`, and for any timeout/garbled ones re-probe via a single capped `curl` to classify quickly.

## 6. Known issues & fixes (hard-won — check these first)
### 6.1 🔴 `ROCM_QUICK_REDUCE_QUANTIZATION=INT4` garbles long/complex generations
INT4-quantized TP all-reduce accumulates error over long agentic prompts → output degenerates into token-soup / runaway (no EOS) / native tool markup (`]<]minimax[>[`) leaking into content / typo'd enum values / wrong arg types. Short/simple calls look fine, so it hides.
- **Impact:** silently tanks ToolCalls-Match, Schema-Accuracy, Scenario-Check, and causes 1800s pytest timeouts.
- **Fix:** do NOT pass `ROCM_QUICK_REDUCE_QUANTIZATION` (the bundled launch defaults `QR_QUANT=none`). Verified: removing it recovered 9/9 Basic-Check tool-call FNs (90%→~99%), 3/3 schema errors (→~100%), and Scenario-Check (0%→100%).
- To reproduce the bug: `QR_QUANT=INT4 bash scripts/launch_sglang_tp4.sh`.

### 6.2 aiter/MXFP8 `HIP illegal memory access` (Triton Code 700)
`_mxfp8_linear_kernel` / MoE router crash under sustained load (~20-25 reqs @ conc 2; instant @ conc 8 or mixed load). HIP is async so the reported frame drifts. **Mitigation:** run at low concurrency and use the resilient runner (auto-restart + `--incremental`). This is an engine/build defect — flag for the SGLang team, don't try to fully fix.

### 6.3 detokenizer hang
Occasional forward stall + `503 detokenizer 20s no response`; scheduler watchdog (300s) eventually aborts and the runner restarts. Just let it recover.

### 6.4 Updating to newer sglang code that needs newer aiter
New M3 commits may `from aiter.ops.shuffle import moe_shuffle_scale` (gated by `SGLANG_M3_MOE_AITER=1`). On gfx950 that's a pure-Python wrapper over the existing `shuffle_scale` and can be backported, BUT the compiled `torch.ops.aiter.fused_moe` is also newer-ABI → `RuntimeError: Scales should have the same dtype!`. So the aiter MoE path needs a **full aiter rebuild** (heavy, composable_kernel submodule may fail). If you only need Format/Basic results (which don't depend on MoE backend), run with `MOE_AITER=0` (triton MoE) instead. sglang is editable so pure-Python sglang changes need only a restart, no rebuild.

## 7. Report
Write `QC_REPORT.md` with: config, per-metric table vs baseline, the 150→(passed/skipped/transient/INT4/genuine) breakdown, per-genuine-failure root cause + fix, and the §6 stability findings. Distinguish "as-submitted config" numbers from "verified after fix" numbers, and state clearly whether a single clean full run was completed.

## Baselines (manual)
Query-Success 100% · ToolCalls-Match 98.80% · Schema-Accuracy 98.93% · Error-Only-Reasoning 0% · Language-Following 100% · Scenario-Check 100%. Format text 150 cases; 4 json_object cases auto-skip on M3.
