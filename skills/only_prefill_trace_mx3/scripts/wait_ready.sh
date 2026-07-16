#!/usr/bin/env bash
# Poll /health until the ATOM server is up, then print key engine kwargs to confirm config.
set -u
PORT=${PORT:-8000}
OUT=${OUT:?set OUT}
LOG="$OUT/logs/server.log"

for i in $(seq 1 60); do
  h=$(curl -s "http://localhost:$PORT/health" 2>/dev/null)
  if [ -n "$h" ]; then echo "HEALTH after ${i}0s: $h"; break; fi
  sleep 10
done

echo "--- engine kwargs (verify TP / chunk / kv / profiler) ---"
grep -m1 "Engine kwargs" "$LOG" | grep -o \
  -e "'tensor_parallel_size': [0-9]*" \
  -e "'max_num_batched_tokens': [0-9]*" \
  -e "'kv_cache_dtype': '[^']*'" \
  -e "'max_model_len': [0-9]*" \
  -e "'enable_prefix_caching': [A-Za-z]*" \
  -e "'attn_prefill_chunk_size': [0-9]*" \
  -e "'torch_profiler_dir': '[^']*'"
