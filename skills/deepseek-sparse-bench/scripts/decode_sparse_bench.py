#!/usr/bin/env python3
# DeepSeek-V4 Sparse Decode Attention benchmark (aiter / gfx950).
# Operator: aiter.mla.mla_decode_fwd (sparse MLA decode, topk selection).
#
# Command (run from the script's directory):
#   PYTORCH_ALLOC_CONF=expandable_segments:True \   # env var: avoid fragmentation OOM
#   python \                                          # run with python
#   decode_sparse_bench.py \                          # script (run from its directory)
#   --batch 256 \                                     # concurrent decode requests
#   --context 65536 \                                 # KV cache length per request
#   --topk 1024 \                                     # sparse budget per token (V4-Pro = 1024)
#   --dtype fp8                                       # data type (bf16 / fp8)
import argparse, torch
import aiter
from aiter import dtypes

def parse():
    p = argparse.ArgumentParser(description="DeepSeek-V4 sparse decode attention bench")
    p.add_argument("--batch", type=int, required=True, help="concurrent decode requests")
    p.add_argument("--context", type=int, required=True, help="KV cache length per request")
    p.add_argument("--topk", type=int, default=1024, help="sparse budget per token (V4-Pro=1024)")
    p.add_argument("--dtype", choices=["bf16", "fp8"], default="bf16", help="q & kv dtype")
    p.add_argument("--heads", type=int, default=128)
    p.add_argument("--kv-lora-rank", type=int, default=512)
    p.add_argument("--qk-rope", type=int, default=64)
    p.add_argument("--max-split", type=int, default=32)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

if __name__ == "__main__":
    a = parse()
    dev = "cuda"
    H, KVLORA, QROPE = a.heads, a.kv_lora_rank, a.qk_rope
    QK, VH, PS, MS = KVLORA + QROPE, KVLORA, 1, a.max_split
    nhead_kv, QLEN = 1, 1
    K = min(a.topk, a.context)
    sm = 1.0 / (QK ** 0.5)
    dtype = dtypes.bf16 if a.dtype == "bf16" else dtypes.fp8
    kvtype = dtype
    num_page = 65536 * 32
    bs = a.batch
    torch.manual_seed(a.seed); g = torch.Generator(device=dev).manual_seed(a.seed)

    seq_kv = torch.full((bs,), a.context, dtype=torch.int)
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int); kv_indptr[1:] = torch.cumsum(seq_kv, 0)
    qo_indptr = torch.arange(0, bs + 1, dtype=torch.int) * QLEN
    kv_last = torch.ones(bs, dtype=torch.int)
    kv_indices = torch.randint(0, num_page, (kv_indptr[-1].item(),), dtype=torch.int)  # block table
    kv_indptr = kv_indptr.cuda(); qo_indptr = qo_indptr.cuda(); kv_last = kv_last.cuda()
    kv_indices = kv_indices.cuda()
    total_q = qo_indptr[-1].item()

    kv_buffer = torch.randn((num_page * PS, 1, QK), dtype=torch.bfloat16, device=dev).mul_(0.5)
    q = torch.randn((total_q, H, QK), dtype=torch.bfloat16, device=dev).mul_(0.5)

    # metadata (work-stealing splits / reduce maps)
    info = aiter.get_mla_metadata_info_v1(bs, QLEN, H, dtype, kvtype, is_sparse=True,
                                          fast_mode=True, num_kv_splits=MS)
    md = [torch.empty(sz, dtype=tp, device=dev) for (sz, tp) in info]
    wmd, wip, wis, rip, rfm, rpm = md
    aiter.get_mla_metadata_v1(qo_indptr, kv_indptr, kv_last, H // nhead_kv, nhead_kv, True,
                              wmd, wis, wip, rip, rfm, rpm, page_size=PS, kv_granularity=max(PS, 16),
                              max_seqlen_qo=1, uni_seqlen_qo=1, fast_mode=True,
                              max_split_per_batch=MS, topk=a.topk, dtype_q=dtype, dtype_kv=kvtype)

    # sparse selection -> global slot indices (page_size=1: slot = block_table[kv_indptr[req] + pos])
    tok = torch.randint(0, a.context, (total_q, K), generator=g, device=dev, dtype=torch.int64)
    base = kv_indptr[:bs].to(torch.int64).repeat_interleave(QLEN).unsqueeze(1)   # per-token req base
    cidx = kv_indices[(base + tok).reshape(-1)].to(torch.int32)                  # [total_q*K]

    out = torch.empty((total_q, H, VH), dtype=torch.bfloat16, device=dev)
    if dtype == dtypes.fp8: q = q.to(dtypes.fp8)
    if kvtype == dtypes.fp8: kv_buffer = kv_buffer.to(dtypes.fp8)
    qs = torch.ones([1], device=dev); ks = torch.ones([1], device=dev)
    kw = dict(num_kv_splits=MS, work_meta_data=wmd, work_indptr=wip, work_info_set=wis,
              reduce_indptr=rip, reduce_final_map=rfm, reduce_partial_map=rpm)
    if dtype == dtypes.fp8 or kvtype == dtypes.fp8: kw.update(q_scale=qs, kv_scale=ks)
    args = (q, kv_buffer.view(num_page, PS, nhead_kv, QK), out, qo_indptr, kv_indptr,
            cidx, kv_last, 1, PS, nhead_kv, sm)

    import aiter.mla
    for _ in range(a.warmup): aiter.mla.mla_decode_fwd(*args, **kw)
    torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
    for _ in range(a.iters): aiter.mla.mla_decode_fwd(*args, **kw)
    e.record(); torch.cuda.synchronize()
    lat = s.elapsed_time(e) / a.iters * 1000  # us
    tflops = 2.0 * total_q * K * H * (QK + VH) / (lat * 1e-6) / 1e12
    print(f"[DECODE] batch={a.batch} context={a.context} topk={a.topk} H={a.heads} dtype={a.dtype} "
          f"| latency={lat/1000:.4f}ms TFLOPS={tflops:.0f}")
