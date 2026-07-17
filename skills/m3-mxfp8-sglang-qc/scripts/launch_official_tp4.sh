#!/usr/bin/env bash
# MiniMax-M3-MXFP8 官方 upstream SGLang 启动脚本 (AMD MI355X gfx950, TP4, 卡 4-7)
# 严格基于官方 cookbook: docs_new/src/snippets/configs/MiniMaxAI/minimax-m3.jsx 的 mi355x cell
#   env:   SGLANG_USE_AITER=1
#   flags: --trust-remote-code --reasoning-parser auto --tool-call-parser auto
#          --quantization mxfp8 --dtype bfloat16 --chunked-prefill-size 8192
#          --mem-fraction-static 0.80
# 官方 recipe 为 --tp 8;本次按用户要求用卡 4-7 => --tp 4。
# 仅额外加 --context-length 1048576(=模型原生 max_position,让 512K/1M 长上下文 format 用例不被拒)。
# 不加任何 ATOM 分支 env(SGLANG_MINIMAX_M3_*)、不加 --api-key / --disable-cuda-graph。
#
# ⚠️ upstream bug 绕过:当前 HEAD(commit 0663ebc783, PR #28715)在 AMD/HIP 上,
#   _minimax_m3_overrides 于 enable_aiter_allreduce_fusion=False(默认)时声明
#   overrides["disable_custom_all_reduce"]=True,而该字段无 resolvable=True,
#   validate_declarations 直接抛 ValueError => 默认 recipe 无法启动。
#   本脚本加 --enable-aiter-allreduce-fusion 跳过该声明分支以启动(不改源码)。
set -x

TP=${TP:-4}
PORT=${PORT:-8043}
GPUS=${GPUS:-4,5,6,7}
MEM_FRAC=${MEM_FRAC:-0.80}
CTX=${CTX:-1048576}
SNAP=${SNAP:-/home/lucy/work/models--MiniMaxAI--MiniMax-M3-MXFP8/snapshots/c5454eb03678d8710e54a4e0fc681b9f3b4a3dba}
LOG=${LOG:-/home/lucy/work/atom_sglang/results-tp4-sglang-mxfp8-upstream-20260717/logs/server_${PORT}.log}

cd /sgl-workspace/sglang

# 操作层防护(不影响推理正确性):本容器 cgroup pids.max=2048 只读、无法抬高;
# 384 核机器上 4 个 TP rank 各库线程池会撑爆 2048 => pthread_create EAGAIN (Gloo thread.cpp:241) abort。
# 故把所有会按核数起池的库线程数硬压低,并开 CPU 亲和性(每 rank 只看到子集核)。
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-8}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export GOMP_SPINCOUNT=0

# 可选:验证"配置可解"用例时开启(默认不设 = 忠实官方 recipe)
EXTRA=""
[ -n "${API_KEY:-}" ] && EXTRA="$EXTRA --api-key ${API_KEY}"
[ "${CACHE_REPORT:-0}" = "1" ] && EXTRA="$EXTRA --enable-cache-report"

SGLANG_USE_AITER=1 SGLANG_SET_CPU_AFFINITY=1 \
HIP_VISIBLE_DEVICES=${GPUS} \
python -m sglang.launch_server --model-path "${SNAP}" --trust-remote-code \
  --reasoning-parser auto --tool-call-parser auto \
  --tp-size "${TP}" \
  --quantization mxfp8 --dtype bfloat16 \
  --enable-aiter-allreduce-fusion \
  --chunked-prefill-size 8192 \
  --mem-fraction-static "${MEM_FRAC}" \
  --context-length "${CTX}" \
  ${EXTRA} \
  --host 0.0.0.0 --port "${PORT}" \
  > "${LOG}" 2>&1 &

echo "launched pid=$! log=${LOG} tp=${TP} gpus=${GPUS} ctx=${CTX}"
