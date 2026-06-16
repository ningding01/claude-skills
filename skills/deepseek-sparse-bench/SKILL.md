---
name: deepseek-sparse-bench
description: Benchmark DeepSeek-V4 sparse attention (prefill & decode) on AMD aiter / gfx950. Use when the user wants to measure latency/TFLOPS of pa_sparse_prefill_opus (sparse prefill) or aiter.mla.mla_decode_fwd (sparse decode), sweep N / batch / context / topk / dtype, or compare against full MLA. Also covers aligning a NVIDIA (FlashMLA) run on B300.
---

# DeepSeek-V4 Sparse Attention Benchmark

Benchmark sparse attention for DeepSeek-V4 on AMD aiter (gfx950 / MI355X).

## Operators
- **Prefill sparse**: `pa_sparse_prefill_opus` (aiter, develop JIT C++; bf16/fp16 only).
- **Decode sparse**: `aiter.mla.mla_decode_fwd` (production asm; bf16 + fp8).
- **Full baseline**: `aiter.mla.mla_prefill_fwd`.
- Selection (which KV) is upstream `top_k_per_row` + indexer; the attention op only gathers indices.

## DeepSeek-V4-Pro config
H=128, num_kv_heads=1 (MQA), head_dim/kv_lora_rank=512, qk_rope_head_dim=64, **index_topk=1024**.
(`q_lora_rank=1536` and `o_lora_rank=1024` are projection ranks — NOT used by the attention kernel.)
aiter's default topk is 2048; V4-Pro real value is 1024.

## Prerequisites
- AMD gfx950, aiter installed (importable: `import aiter`).
- Prefill kernel needs the int64 offset fix in `pa_sparse_prefill_opus.h` (lines ~1272/1750:
  `qo_gmem_offset` cast to int64) to support N>=32768; without it N>=65536 hits a memory fault.
  After editing, delete the JIT cache (`aiter/jit/module_pa_sparse_prefill_opus.so` and
  `aiter/jit/build/module_pa_sparse_prefill_opus/`) to trigger a rebuild.
- Large N needs `PYTORCH_ALLOC_CONF=expandable_segments:True` (avoids fragmentation OOM).

## Usage

Prefill (sweep `--num-tokens` over 1024/4096/16384/65536/262144):
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True python scripts/prefill_sparse_bench.py \
  --num-tokens 262144 --heads 128 --head-dim 512 --topk 1024 --dtype bf16
# -> [SPARSE] N=262144 H=128 D=512 topk=1024 dtype=bf16 | nnz=... latency=..ms TFLOPS=..
```

Decode (sweep `--batch` / `--context` / `--dtype`):
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True python scripts/decode_sparse_bench.py \
  --batch 256 --context 65536 --topk 1024 --dtype fp8
# -> [DECODE] batch=256 context=65536 topk=1024 H=128 dtype=fp8 | latency=..ms TFLOPS=..
```

Run from this skill's directory (aiter is an installed package; no `cd` into the aiter tree needed).
Use `--topk 2048` for the aiter default instead of the V4-Pro 1024.

## Methodology (baked into the scripts)
- Prefill: causal sparse, budget min(i+1, topk), scattered per-token seeded indices, pool=N.
- Decode: each request 1 query token attending topk selected KV; scattered seeded indices.
- Timing: CUDA events, warmup=3, iters=20/30, latency in ms.
- TFLOPS: prefill `4*H*nnz*D`; decode `2*total_q*topk*H*(qk+v)`. (Latency is the robust metric;
  full-attention TFLOPS is inflated by causal tile-skipping, so the scripts report sparse only.)

## Key findings (reference)
- Prefill sparse latency is linear in N (O(N·topk)); full is quadratic (O(N²)). At topk=1024,
  crossover vs full ≈ N≈43K; at N=262K sparse ~5.7× faster than full. (topk=2048: crossover ~75K.)
- Decode latency is flat in context length (capped at topk); fp8 gives ~1.5× only at batch>=128
  (memory-bound regime); small batch is overhead-bound (~72µs floor).
- The "scatter gather penalty" is driven by cross-token cache reuse, NOT physical contiguity.
- Sparse kernels (AMD aiter & NVIDIA FlashMLA) do NOT use an in-kernel paged block table; paging is
  pre-resolved to flat global indices upstream.

## References
- `references/JIRA_sparse_attention_bench.md` — ready-to-paste commands for tickets.
- `references/B300_nvidia_sparse_alignment.md` — how to run an aligned NVIDIA FlashMLA bench on B300.
- `references/*_topk1024.csv` — measured V4-Pro (topk=1024) results on MI355X.
