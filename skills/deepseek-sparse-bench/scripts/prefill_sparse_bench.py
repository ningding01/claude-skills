#!/usr/bin/env python3
# DeepSeek-V4 Sparse Prefill Attention benchmark (aiter / gfx950).
# Operator: pa_sparse_prefill_opus
#
# Command (run from the script's directory):
#   PYTORCH_ALLOC_CONF=expandable_segments:True \   # env var: avoid fragmentation OOM (needed for large N)
#   python \                                          # run with python
#   prefill_sparse_bench.py \                         # script (run from its directory)
#   --num-tokens 262144 \                             # N = query token count (prefill seqlen)
#   --heads 128 \                                     # H = number of attention heads
#   --head-dim 512 \                                  # D = head dim (= kv_lora_rank)
#   --topk 1024 \                                     # sparse budget per token (V4-Pro = 1024)
#   --dtype bf16                                      # data type (bf16 / fp16)
import argparse, math, torch

def parse():
    p = argparse.ArgumentParser(description="DeepSeek-V4 sparse prefill attention bench")
    p.add_argument("--num-tokens", type=int, required=True, help="N = query token count (prefill seqlen)")
    p.add_argument("--heads", type=int, default=128, help="num attention heads H")
    p.add_argument("--head-dim", type=int, default=512, help="head dim D (= kv_lora_rank); kernel compiled for 512")
    p.add_argument("--topk", type=int, default=1024, help="sparse budget per token (V4-Pro=1024)")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def sparse_csr(N, K, seed, dev):
    g = torch.Generator(device=dev).manual_seed(seed)
    klen = torch.clamp(torch.arange(1, N + 1, device=dev), max=K).to(torch.int32)  # causal budget min(i+1,K)
    ip = torch.zeros(N + 1, dtype=torch.int32, device=dev); ip[1:] = torch.cumsum(klen, 0)
    nnz = int(ip[-1]); ix = torch.empty(nnz, dtype=torch.int32, device=dev)
    for i in range(min(K, N)):                                # causal warmup rows: contiguous [0,i]
        s = int(ip[i]); ix[s:s + i + 1] = torch.arange(i + 1, device=dev, dtype=torch.int32)
    if N > K:                                                 # steady rows: K scattered from [0,i]
        rows = torch.arange(K, N, device=dev); s = int(ip[K])
        r = torch.rand((N - K, K), generator=g, device=dev)
        ix[s:] = (r * (rows + 1).unsqueeze(1).float()).to(torch.int32).reshape(-1)
    return ip, ix, nnz

def bench_sparse(a, dev, dt):
    from aiter.ops.pa_sparse_prefill_opus import pa_sparse_prefill_opus
    N, H, D, K = a.num_tokens, a.heads, a.head_dim, a.topk
    scale = 1.0 / math.sqrt(D)
    torch.manual_seed(a.seed)
    q = (torch.randn(N, H, D, device=dev, dtype=torch.float32) * 0.5).to(dt)
    ukv = (torch.randn(N, D, device=dev, dtype=torch.float32) * 0.5).to(dt)
    ip, ix, nnz = sparse_csr(N, K, a.seed, dev)
    kv = (torch.randn(1, D, device=dev, dtype=torch.float32) * 0.5).to(dt)
    ipe = torch.zeros(N + 1, dtype=torch.int32, device=dev); ixe = torch.zeros(0, dtype=torch.int32, device=dev)
    sink = torch.randn(H, device=dev, dtype=torch.float32) * 0.25
    inp = dict(q=q, unified_kv=ukv, kv_indices_prefix=ix, kv_indptr_prefix=ip, kv=kv,
               kv_indices_extend=ixe, kv_indptr_extend=ipe, attn_sink=sink)
    for _ in range(a.warmup): pa_sparse_prefill_opus(**inp, softmax_scale=scale)
    torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
    for _ in range(a.iters): pa_sparse_prefill_opus(**inp, softmax_scale=scale)
    e.record(); torch.cuda.synchronize()
    lat = s.elapsed_time(e) / a.iters * 1000  # us
    tflops = 4.0 * H * nnz * D / (lat * 1e-6) / 1e12
    return nnz, lat, tflops

if __name__ == "__main__":
    a = parse()
    dev = "cuda"; dt = torch.bfloat16 if a.dtype == "bf16" else torch.float16
    nnz, sl, stf = bench_sparse(a, dev, dt)
    print(f"[SPARSE] N={a.num_tokens} H={a.heads} D={a.head_dim} topk={a.topk} dtype={a.dtype} "
          f"| nnz={nnz} latency={sl/1000:.4f}ms TFLOPS={stf:.0f}")
