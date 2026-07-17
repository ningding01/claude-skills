---
name: m3-mxfp8-sglang-qc
description: Run MiniMax-M3-MXFP8 supplier quality inspection (Format Correctness + Basic Check) against a standalone SGLang server on AMD ROCm (MI355X/gfx950, TP4). Use when asked to QC / 质检 / verify an M3-MXFP8 SGLang deployment with the MiniMax-Provider-Verifier, launch the server for it, diagnose failures, or reproduce the known ROCM_QUICK_REDUCE_QUANTIZATION=INT4 garbled-output issue.
---

# MiniMax-M3-MXFP8 · SGLang 质量检测 Skill

End-to-end supplier QC for MiniMax-M3-MXFP8 on **standalone SGLang + AMD ROCm**, aligned to
`M3 External Provider Quality Inspection Manual`. Covers the two self-inspection items that are
deterministic and compute-cheap: **Format Correctness (text)** and **Inference Quality → Basic Check**.

Bundled scripts (in `scripts/`):
- `launch_sglang_tp4.sh` — hardened launch (all fixes baked in; env-overridable)
- `run_basic_check_resilient.sh` — crash-resilient Basic Check (auto-restart + `--incremental`)

## 0. Prerequisites
- SGLang installed **editable** (`pip show sglang` → "Editable project location"); branch supporting `minimax_m3_vl`.
- aiter present under `/sgl-workspace/aiter` (JIT-compiled kernels).
- `MiniMax-Provider-Verifier` checkout (has `verify.py`, `sample.jsonl` 102 samples, `m3_format_check/m3_text_tests.py` 150 cases).
- Deps: `pip install pytest pytest-xdist pytest-timeout httpx jsonschema loguru megfile tqdm`.
- GPUs free (check `rocm-smi --showmemuse`); pick idle cards.

## 1. Pre-flight (do these BEFORE launching — they are the top failure causes)
1. **Raise cgroup pids.max** (it defaults to ~2048 and *reverts*; a 768-core box explodes past it → `RuntimeError: can't start new thread` / OpenBLAS `pthread_create failed` crash):
   ```bash
   echo 200000 > /sys/fs/cgroup/pids.max
   ```
   Re-apply before EVERY (re)launch. For your own diagnostic `python3 -c`/`git` commands also `export OPENBLAS_NUM_THREADS=1` or they segfault the same way.
2. Confirm model snapshot path, free GPUs, and free port.

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

Wait for health (M3 = 31 shards, ~7 min):
```bash
for i in $(seq 1 60); do [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8043/health)" = 200 ] && break; sleep 10; done
```

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
