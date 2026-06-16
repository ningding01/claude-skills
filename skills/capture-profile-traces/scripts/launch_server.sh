#!/usr/bin/env bash
# Launch an SGLang or vLLM server with the torch-profiler dir set, so capture_traces.py
# can drive /start_profile + /stop_profile. Edit the vars or pass them as env.
#
#   FRAMEWORK=sglang MODEL=/path/to/model TP=8 TRACE_DIR=/abs/traces PORT=8080 \
#     EXTRA="--quantization mxfp8 --chunked-prefill-size 8192 --trust-remote-code" \
#     bash launch_server.sh
#
# Keep traces lean: WITH_STACK/RECORD_SHAPES off. For decode you want CUDA graphs ON
# (default) — that's the real serving path; the analysis skill knows decode is graphed.
set -euo pipefail

FRAMEWORK="${FRAMEWORK:-sglang}"
MODEL="${MODEL:?set MODEL=/path/to/model}"
TP="${TP:-8}"
TRACE_DIR="${TRACE_DIR:?set TRACE_DIR=/abs/path/traces}"
PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"
EXTRA="${EXTRA:-}"

mkdir -p "$TRACE_DIR"

if [[ "$FRAMEWORK" == "sglang" ]]; then
  export SGLANG_TORCH_PROFILER_DIR="$TRACE_DIR"
  export SGLANG_PROFILE_WITH_STACK=0
  export SGLANG_PROFILE_RECORD_SHAPES=0
  # SGLANG_USE_AITER=1 etc. can be added via the environment before calling this script.
  exec sglang serve --model-path "$MODEL" --tp "$TP" \
       --host "$HOST" --port "$PORT" $EXTRA
elif [[ "$FRAMEWORK" == "vllm" ]]; then
  export VLLM_TORCH_PROFILER_DIR="$TRACE_DIR"
  # prefix caching lets the decode-only capture skip prefill (cache hit).
  exec vllm serve "$MODEL" --tensor-parallel-size "$TP" \
       --host "$HOST" --port "$PORT" --enable-prefix-caching $EXTRA
else
  echo "unknown FRAMEWORK=$FRAMEWORK (use sglang|vllm)"; exit 1
fi
