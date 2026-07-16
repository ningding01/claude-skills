#!/usr/bin/env bash
# Launch ATOM OpenAI server for MiniMax-M3 (MXFP8) pure-prefill trace capture.
# NO EAGLE / speculative decode. Torch profiler enabled.
# Params (env): TP, GPUS, PORT, OUT, MODEL, UTIL, MAXLEN, CHUNK.
set -u
TP=${TP:-4}
GPUS=${GPUS:-4,5,6,7}
PORT=${PORT:-8000}
OUT=${OUT:?set OUT to the run output dir}
MODEL=${MODEL:-/projects/models/MiniMax-M3-MXFP8}
UTIL=${UTIL:-0.8}
MAXLEN=${MAXLEN:-100000}
CHUNK=${CHUNK:-8192}          # keep IDENTICAL across configs you compare (= TP8 chunk prefill)

PROF="$OUT/profile_out"
LOG="$OUT/logs/server.log"
mkdir -p "$PROF" "$OUT/logs"

export HIP_VISIBLE_DEVICES="$GPUS" CUDA_VISIBLE_DEVICES="$GPUS"
export ATOM_FORCE_ATTN_TRITON=1 PYTHONPATH=/app/ATOM
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_PROFILER_MORE=1   # REQUIRED: record_shapes + with_stack

python3 -m atom.entrypoints.openai_server \
  --model "$MODEL" \
  --tensor-parallel-size "$TP" \
  --server-port "$PORT" \
  --trust-remote-code \
  --gpu-memory-utilization "$UTIL" \
  --block-size 128 \
  --max-model-len "$MAXLEN" \
  --max-num-seqs 128 \
  --max-num-batched-tokens "$CHUNK" \
  --kv_cache_dtype fp8 \
  --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"]}' \
  --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
  --torch-profiler-dir "$PROF" \
  > "$LOG" 2>&1 &
echo "server pid $!  (TP=$TP GPUS=$GPUS PORT=$PORT chunk=$CHUNK) -> $LOG"
