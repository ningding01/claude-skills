#!/usr/bin/env bash
# Crash-resilient Basic Check runner for MiniMax-M3 on SGLang.
# verify.py writes results only at the END of each pass and marks server-crash
# requests as "failed", so it always writes 102 lines. We therefore count
# SUCCESSES (not lines) and loop with --incremental, auto-restarting the server
# after each aiter/MXFP8 crash until all samples succeed.
#
# Required env:
#   VERIFIER = path to MiniMax-Provider-Verifier checkout (verify.py, sample.jsonl)
#   OUTDIR   = directory for results/summary
#   LAUNCH   = path to launch_sglang_tp4.sh
#   M3_MODEL = served model name/path
# Optional: PORT(8043) CONC(2) MAXTOK(2048; verify.py force-overrides M3 to 40960) MAX_ROUNDS(20)
set -u
: "${VERIFIER:?set VERIFIER}"; : "${OUTDIR:?set OUTDIR}"; : "${LAUNCH:?set LAUNCH}"; : "${M3_MODEL:?set M3_MODEL}"
PORT=${PORT:-8043}
RES="$OUTDIR/basic_check_results.jsonl"; SUM="$OUTDIR/basic_check_summary.json"
CONC=${CONC:-2}; MAXTOK=${MAXTOK:-2048}; MAX_ROUNDS=${MAX_ROUNDS:-20}
mkdir -p "$OUTDIR"

health(){ curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" 2>/dev/null; }
ensure_server(){
  [ "$(health)" = "200" ] && return 0
  echo "[runner] server down -> restart"
  pkill -9 -f 'sglang::' 2>/dev/null; pkill -9 -f 'launch_server' 2>/dev/null; sleep 3
  echo 200000 > /sys/fs/cgroup/pids.max 2>/dev/null   # keep pids.max high (it reverts to 2048)
  bash "$LAUNCH" >/dev/null 2>&1
  for _ in $(seq 1 60); do [ "$(health)" = "200" ] && { echo "[runner] up"; return 0; }; sleep 10; done
  echo "[runner] server failed to come up"; return 1
}
done_count(){ [ -f "$RES" ] && grep -c '"status": "success"' "$RES" 2>/dev/null || echo 0; }

for r in $(seq 1 "$MAX_ROUNDS"); do
  ensure_server || break
  echo "[runner] round $r  success=$(done_count)/102"
  ( cd "$VERIFIER" && python3 verify.py sample.jsonl \
      --model "$M3_MODEL" --base-url "http://127.0.0.1:${PORT}/v1" --api-key dummy \
      --concurrency "$CONC" --timeout 600 --retries 1 \
      --extra-body "{\"max_tokens\":$MAXTOK}" --incremental \
      --output "$RES" --summary "$SUM" ) >> "$OUTDIR/basic_check.out" 2>&1
  dc=$(done_count); echo "[runner] round $r done success=$dc/102"
  [ "$dc" -ge 102 ] && { echo "[runner] ALL DONE"; break; }
done
echo "[runner] final success=$(done_count)/102"
