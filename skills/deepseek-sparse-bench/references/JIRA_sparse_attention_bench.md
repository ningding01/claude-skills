# DeepSeek-V4 Sparse Attention Benchmark (aiter / gfx950 MI355X)

Config (DeepSeek-V4-Pro): num_attention_heads=128, num_kv_heads=1, head_dim(kv_lora_rank)=512,
qk_rope_head_dim=64, **index_topk=1024**. Operators:
- Prefill sparse: `pa_sparse_prefill_opus`  | full baseline: `aiter.mla.mla_prefill_fwd`
- Decode  sparse: `aiter.mla.mla_decode_fwd`

Prereq: aiter installed; run from the aiter source tree; set `PYTORCH_ALLOC_CONF=expandable_segments:True`.
(Prefill kernel needs the int64 offset fix in `pa_sparse_prefill_opus.h` to support N>=32768.)

Scripts: `prefill_sparse_bench.py`, `decode_sparse_bench.py`

---

## Prefill — Command to test

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True python prefill_sparse_bench.py \
  --num-tokens 262144 --heads 128 --head-dim 512 --topk 1024 --dtype bf16
```
Change `--num-tokens` to sweep N: 1024 / 4096 / 16384 / 65536 / 262144.

Output example:
```
[SPARSE] N=262144 H=128 D=512 topk=1024 dtype=bf16 | nnz=267911680 latency=78.4420ms TFLOPS=895
```

---

## Decode — Command to test

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True python decode_sparse_bench.py \
  --batch 256 --context 65536 --topk 1024 --dtype fp8
```
Change `--batch` / `--context` / `--dtype` (bf16|fp8) per case.

Output example:
```
[DECODE] batch=256 context=65536 topk=1024 H=128 dtype=fp8 | latency=0.0721ms TFLOPS=1013
```

Run from the directory containing the script (aiter is an installed package, importable from anywhere; no `cd` into the aiter tree needed).
`PYTORCH_ALLOC_CONF=expandable_segments:True` is required for large N (avoids fragmentation OOM); harmless for small configs.

---

## Notes
- `--topk 1024` is the V4-Pro value; pass `--topk 2048` for the aiter default.
- Prefill: `latency`/`TFLOPS` reliable; full baseline is for scaling comparison (full latency reliable, full TFLOPS would be inflated by causal masking so not reported here).
- Decode: q & kv both use `--dtype` (fp8 = q+kv fp8). fp8 speedup only materializes at batch>=128 (memory-bound regime).
