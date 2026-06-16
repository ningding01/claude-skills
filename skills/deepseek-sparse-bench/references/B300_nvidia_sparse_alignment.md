# Instructions: Benchmark NVIDIA sparse MLA prefill on B300, aligned with the AMD `pa_sparse_prefill_opus` benchmark

Goal: produce a NVIDIA-side sparse prefill attention benchmark on B300 (Blackwell, sm100) whose
numbers are directly comparable to an AMD (gfx950) benchmark of `pa_sparse_prefill_opus`. Match the
DeepSeek-V4-Pro config and the same measurement methodology, then report the same metrics.

You (the B300 agent) should: install the kernel, write a small self-contained benchmark script
(mirroring the AMD one described below), run the N sweep, and print one line per config.

---

## 1. The NVIDIA operator to benchmark

`flash_mla_sparse_fwd` from DeepSeek **FlashMLA** (the sparse MLA prefill kernel; this is exactly
what vLLM uses for DeepSeek-V4 sparse attention on NVIDIA).

- vLLM call site (reference): `vllm/v1/attention/backends/mla/flashmla_sparse.py`
  ```python
  topk_indices = topk_indices.view(num_tokens, 1, -1)
  output = flash_mla_sparse_fwd(q, kv_c_and_k_pe_cache, topk_indices, softmax_scale)[0]
  ```
- Import (either works depending on install):
  ```python
  from flash_mla import flash_mla_sparse_fwd            # standalone FlashMLA package
  # or, if using vLLM's vendored copy:
  from vllm.third_party.flashmla.flash_mla_interface import flash_mla_sparse_fwd
  ```
- Install: build/install DeepSeek FlashMLA with Blackwell (sm100) support, e.g.
  `pip install git+https://github.com/deepseek-ai/FlashMLA.git` (use a commit/branch that supports
  sm100/B300 BF16 sparse prefill), or install a vLLM build that vendors it. Verify the import works.

### Signature / tensor shapes (MQA 576/512 absorbed form)
- `q`: `[num_tokens, num_heads, head_dim]`, **head_dim = kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576**, bf16.
  On Blackwell the prefill kernel needs num_heads padded to a multiple of 128 (V4 uses 128, already aligned).
- `kv_c_and_k_pe_cache`: the flat KV cache, shape `[num_slots, 1, 576]`, bf16.
- `topk_indices`: `[num_tokens, 1, topk]`, **int32**, global flat indices into `kv_c_and_k_pe_cache`
  (the already-resolved selected KV slots; padded entries = -1).
- `softmax_scale`: `1/sqrt(576)`.
- Returns `(output, lse)`; `output`: `[num_tokens, num_heads, v_head_dim=512]`.

---

## 2. Config to match (DeepSeek-V4-Pro)

| param | value |
|---|---|
| num_heads (H) | 128 |
| num_kv_heads | 1 (MQA) |
| kv_lora_rank | 512 |
| qk_rope_head_dim | 64  → q/kv head_dim = 576, output v_head_dim = 512 |
| **topk** | **1024** (V4-Pro `index_topk`) |
| dtype | bf16 (primary); fp8 only if FlashMLA sparse fp8 prefill is available |
| N (num_tokens) sweep | **1024, 4096, 16384, 65536, 262144** |

---

## 3. Measurement methodology (MUST match the AMD side)

1. **Causal sparse selection, scattered, seeded** — per query token i, budget = `min(i+1, topk)`;
   the selected indices are `topk` random distinct-ish positions from `[0, i]` (use a seeded RNG so
   it is reproducible). For i < topk the budget is just i+1 (selecting all available, contiguous is fine).
   This mirrors the AMD `sparse_csr` generator (see §5).
2. **Pool grows with N**: the KV cache holds N slots (context = N); selection draws from `[0, i]`.
3. **Timing**: CUDA events, `warmup=3`, `iters=20`, latency = elapsed/iters, report in **ms**.
4. **Metric**: `latency_ms` and `TFLOPS`. Use the SAME FLOP convention so numbers line up:
   - total selected pairs `nnz = sum_i min(i+1, topk)`
   - **For cross-vendor comparison report TFLOPS with the AMD convention `4 * H * nnz * 512`**
     (QK+PV both counted at the 512 latent dim). Optionally also report the "true" FlashMLA FLOPs
     `2 * nnz * H * (576 + 512)` separately, but the aligned column must use `4*H*nnz*512`.
5. **Output line format** (one per config), matching the AMD script:
   ```
   [SPARSE-NV] N=<N> H=128 head_dim=576 topk=1024 dtype=bf16 | nnz=<nnz> latency=<ms>ms TFLOPS=<tf>
   ```

---

## 4. AMD reference (what these numbers compare against)

AMD operator: `pa_sparse_prefill_opus` (aiter, gfx950). AMD output line:
```
[SPARSE] N=262144 H=128 D=512 topk=1024 dtype=bf16 | nnz=267911680 latency=78.4420ms TFLOPS=895
```
AMD measured (topk=1024, bf16, H=128, gfx950 MI355X):

| N | nnz | latency (ms) | TFLOPS |
|---:|---:|---:|---:|
| 1,024 | 524,800 | 0.227 | 605 |
| 4,096 | 3,670,528 | 1.162 | 828 |
| 16,384 | 16,253,440 | 4.625 | 921 |
| 65,536 | 66,585,088 | 19.490 | 896 |
| 262,144 | 267,911,680 | 78.442 | 895 |

Note alignment caveats:
- AMD `pa_sparse_prefill_opus` uses head_dim D=512 for both QK and PV (no separate rope dim in the
  absorbed kernel). NVIDIA FlashMLA uses QK=576 / PV=512. So NVIDIA does slightly more QK work per
  element. The `4*H*nnz*512` TFLOPS convention normalizes the *aligned* column; latency is the
  primary apples-to-apples metric (both = "sparse prefill of N tokens, 1024 KV each, H=128, bf16").
- Both use scattered seeded selection; the absolute index *values* differ (random per side) but that
  does not matter for kernel latency.

---

## 5. AMD generator to mirror (so selection pattern matches)

```python
# causal budget min(i+1, K); warmup rows contiguous [0,i]; steady rows K scattered from [0,i]
def sparse_csr(N, K, seed, dev):
    g = torch.Generator(device=dev).manual_seed(seed)
    klen = torch.clamp(torch.arange(1, N+1, device=dev), max=K).to(torch.int32)
    ip = torch.zeros(N+1, dtype=torch.int32, device=dev); ip[1:] = torch.cumsum(klen, 0)
    nnz = int(ip[-1]); ix = torch.empty(nnz, dtype=torch.int32, device=dev)
    for i in range(min(K, N)):
        s = int(ip[i]); ix[s:s+i+1] = torch.arange(i+1, device=dev, dtype=torch.int32)
    if N > K:
        rows = torch.arange(K, N, device=dev); s = int(ip[K])
        r = torch.rand((N-K, K), generator=g, device=dev)
        ix[s:] = (r * (rows+1).unsqueeze(1).float()).to(torch.int32).reshape(-1)
    return ip, ix, nnz
```
For FlashMLA you need a **dense `[N, topk]` int32** index tensor (not CSR). Build it as: row i =
`topk` selected positions from `[0, i]`; for `i+1 < topk` fill the first `i+1` with `arange(i+1)` and
pad the rest with **-1**; for `i+1 >= topk` fill with `floor(rand(topk) * (i+1))`. Then
`topk_indices = idx.view(N, 1, topk)`.

---

## 6. Deliverable

A script `nvidia_sparse_prefill_bench.py` with the same CLI style as the AMD one:
```
python nvidia_sparse_prefill_bench.py --num-tokens 262144 --heads 128 --topk 1024 --dtype bf16
```
that builds q/kv/topk_indices per §1–§5, times `flash_mla_sparse_fwd`, and prints the §3 line.
Run the full N sweep {1024,4096,16384,65536,262144} and put results in a CSV with columns:
`N,heads,head_dim,topk,dtype,nnz,latency_ms,TFLOPS` so it can sit next to the AMD CSV.

Report: the table, plus note your FlashMLA commit, GPU (B300), driver/CUDA version, and whether bf16
(and fp8 if available) sparse prefill is supported on sm100.
