#!/usr/bin/env bash
# SGLang 独立质检启动脚本 (MiniMax-M3-MXFP8, TP4, GPU 4-7)
# 基于用户给定命令,仅新增 --context-length 覆盖 512K/1M 用例。
set -x

TP=${TP:-4}
PORT=${PORT:-8043}
GPUS=${GPUS:-4,5,6,7}
MEM_FRAC=${MEM_FRAC:-0.7}
# context 设为模型上限 1M(text_config.max_position_embeddings=1048576),覆盖 512K 输入 / 512K max_tokens 用例
CTX=${CTX:-1048576}
SNAP=${SNAP:-/home/lucy/work/models--MiniMaxAI--MiniMax-M3-MXFP8/snapshots/c5454eb03678d8710e54a4e0fc681b9f3b4a3dba}
LOG=${LOG:-/tmp/server_${PORT}.log}
# tool/reasoning 解析器:auto 会从 chat_template 检测为 minimax-m3;可用 TOOL_PARSER/REASONING_PARSER 覆盖
TOOL_PARSER=${TOOL_PARSER:-auto}
REASONING_PARSER=${REASONING_PARSER:-auto}
# 质检不依赖 cuda graph 性能;DISABLE_CG=1 可避免 decode graph 捕获阶段并发 JIT 编译 aiter kernel 的竞争崩溃
DISABLE_CG=${DISABLE_CG:-1}
CG_FLAG=""
[ "${DISABLE_CG}" = "1" ] && CG_FLAG="--disable-cuda-graph"

cd /sgl-workspace/sglang

# cgroup pids.max 有限(容器默认 2048)而机器 768 核:限制每进程线程池,防止 warmup 时 pthread_create EAGAIN 崩溃
# (同时已将 /sys/fs/cgroup/pids.max 抬高作双保险)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-32}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-32}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-32}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-32}

# all-reduce quick-reduce 量化精度。QR_QUANT=none(默认)= 完全不设该变量,走全精度 all-reduce
# (INT4 量化 all-reduce 会在长/复杂生成时累积数值误差导致输出退化成乱码)。
QR_QUANT=${QR_QUANT:-none}
if [ "${QR_QUANT}" != "none" ]; then export ROCM_QUICK_REDUCE_QUANTIZATION="${QR_QUANT}"; fi

SGLANG_M3_ENABLE_CUSTOM_AR=1 \
HIP_VISIBLE_DEVICES=${GPUS} \
SGLANG_USE_AITER=1 SGLANG_SET_CPU_AFFINITY=1 SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
SGLANG_MINIMAX_M3_ATOM_PREFILL=1 SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d \
SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ=${TOPK_FREQ:-4} \
SGLANG_MINIMAX_M3_FUSED_SWIGLU_MXFP8=${FUSED_SWIGLU:-1} \
SGLANG_MINIMAX_M3_FUSED_MOE_COMBINE=${FUSED_COMBINE:-1} \
SGLANG_M3_PTPC_DENSE=${PTPC_DENSE:-1} \
SGLANG_M3_MOE_AITER=${MOE_AITER:-1} \
python -m sglang.launch_server --model-path "${SNAP}" --trust-remote-code \
  --reasoning-parser "${REASONING_PARSER}" --tool-call-parser "${TOOL_PARSER}" --tp-size "${TP}" \
  --host 0.0.0.0 --port "${PORT}" --quantization mxfp8 --mem-fraction-static "${MEM_FRAC}" \
  --context-length "${CTX}" \
  --chunked-prefill-size 16384 --max-prefill-tokens 32768 --page-size 64 \
  --attention-backend aiter \
  --moe-runner-backend ${MOE_BACKEND:-aiter} \
  --cuda-graph-backend-prefill disabled \
  ${CG_FLAG} \
  --enable-metrics \
  > "${LOG}" 2>&1 &

echo "launched pid=$! log=${LOG} ctx=${CTX} tool=${TOOL_PARSER} reasoning=${REASONING_PARSER}"
