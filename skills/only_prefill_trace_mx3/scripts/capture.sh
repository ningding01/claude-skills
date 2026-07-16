#!/usr/bin/env bash
# Drive the ATOM server with the prefill workload and grab a short steady-state
# profile window. Starts agent-bench in the background, waits past ramp into steady
# state, fires /start_profile -> sleep WINDOW -> /stop_profile, then stops the bench.
set -u
TP=${TP:-4}
PORT=${PORT:-8000}
OUT=${OUT:?set OUT}
LT=${LT:-/home/agslibadmin/niding/light-trace-benchmark}
MODEL=${MODEL:-/projects/models/MiniMax-M3-MXFP8}
WINDOW=${WINDOW:-1.5}          # profile window seconds (shorten to 1.0 if trace too big)
WARMUP=${WARMUP:-50}           # seconds to wait before capturing (ramp=30 + margin)
NAME=${NAME:-m3-atom-tp${TP}-prefill-con20-profile-seed42}

HERE="$(cd "$(dirname "$0")" && pwd)"
WL="${WL:-$HERE/workload_prefill_shared90_con20.yaml}"

export PYTHONPATH="$LT"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT/logs"

echo "[capture] starting agent-bench (con20 saturated, ISL75k, OSL1)"
rm -rf "$OUT/benchmark-results/$NAME"
python3 -m agentbench.cli agent \
  --server "http://localhost:$PORT" \
  --model "$MODEL" --tokenizer "$MODEL" --gpus "$TP" \
  --workload-config "$WL" \
  --name "$NAME" --data-dir "$OUT/benchmark-results" --dashboard-mode \
  > "$OUT/logs/agentbench.log" 2>&1 &
BENCH=$!

echo "[capture] warmup ${WARMUP}s (into steady state)..."
sleep "$WARMUP"
echo "[capture] last bench line:"; tail -c 300 "$OUT/logs/agentbench.log" | tr '\r' '\n' | tail -1

echo "[capture] /start_profile -> ${WINDOW}s -> /stop_profile"
curl -s -X POST "http://localhost:$PORT/start_profile" | head -c 200; echo
sleep "$WINDOW"
curl -s -X POST "http://localhost:$PORT/stop_profile" | head -c 300; echo

echo "[capture] stopping bench"
kill "$BENCH" 2>/dev/null
pkill -f "agentbench.cli agent" 2>/dev/null
sleep 2
echo "[capture] traces:"; find "$OUT/profile_out" -name '*.pt.trace.json.gz' | sort
