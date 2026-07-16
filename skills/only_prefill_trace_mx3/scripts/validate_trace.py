#!/usr/bin/env python3
"""Validate an M3 pure-prefill ATOM trace before tearing down the server.

Usage: python3 validate_trace.py <rank_0/*.pt.trace.json.gz>

Checks: GPU kernel count, total GPU-busy ms in the window, top kernels, world_size (TP),
a rough per-step estimate (fmha_fwd count / 3 dense layers = #steps), and flags the
paged_attention decode kernel share (expected small; OSL=1 window is prefill-dominated
but not literally 0% decode).
"""
import gzip, json, sys, glob, collections

def main():
    path = sorted(glob.glob(sys.argv[1]))[0]
    d = json.load(gzip.open(path))
    ev = d["traceEvents"]
    ws = d.get("distributedInfo", {}).get("world_size", "?")
    kern = [e for e in ev if e.get("cat") == "kernel" and "dur" in e]
    tot = sum(e["dur"] for e in kern)
    by = collections.defaultdict(lambda: [0, 0.0])
    for e in kern:
        by[e["name"]][0] += 1
        by[e["name"]][1] += e["dur"]

    print(f"file: {path}")
    print(f"TP (world_size): {ws}")
    print(f"GPU kernel events: {len(kern)}   window GPU-busy: {tot/1000:.1f} ms   unique kernels: {len(by)}")

    fmha = sum(c for n,(c,_) in by.items() if "fmha_fwd" in n or "fmha_v3" in n)
    steps = fmha // 3 if fmha else 0   # M3 has 3 dense layers, each 1 full-attn/step
    if steps:
        print(f"~steps in window: {steps} (fmha_fwd {fmha} / 3 dense layers)  -> ~{tot/1000/steps:.1f} ms/step GPU-busy")

    print("\n-- top 12 GPU kernels (ms / calls) --")
    for n,(c,us) in sorted(by.items(), key=lambda x:-x[1][1])[:12]:
        print(f"  {us/1000:8.1f}ms {c:6d}x  {n[:64]}")

    # sanity flags
    want = ("mfma_moe", "allreduce", "index_block_score", "fmha", "paged_attention",
            "qkv_norm_rope", "rmsnorm")
    missing = [w for w in want if not any(w in n for n in by)]
    print("\nexpected-kernel check:", "all present ✓" if not missing else f"MISSING {missing}")
    dec = sum(us for n,(c,us) in by.items() if "decode" in n.lower())
    print(f"decode-named kernel share: {100*dec/tot:.1f}% (OSL=1 window; small is expected)")

if __name__ == "__main__":
    main()
