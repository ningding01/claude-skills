#!/usr/bin/env python3
"""Aggregate InferenceX `benchmark_serving.py` result JSONs (one per concurrency) into a
single CSV following the official InferenceX schema, so a reproduction can sit next to
official data.

Usage:
  python3 gen_csv.py --results-dir DIR --pattern 'm3_*_con{con}.json' \
      --cons 1,2,4,8,16,32,64,128,256 --tp 4 --isl 8192 --osl 1024 \
      --model MiniMax-M3 --hardware mi355x --framework atom --precision fp8 \
      --date 2026-07-07 --out repro.csv
"""
import argparse, csv, glob, json, os

HEADER = ["Model","ISL","OSL","Hardware","Hardware Key","Framework","Precision","TP","Concurrency","Date",
"Throughput/GPU (tok/s)","Output Throughput/GPU (tok/s)","Input Throughput/GPU (tok/s)",
"Mean TTFT (ms)","Median TTFT (ms)","P99 TTFT (ms)","Std TTFT (ms)",
"Mean TPOT (ms)","Median TPOT (ms)","P99 TPOT (ms)","Std TPOT (ms)",
"Mean Interactivity (tok/s/user)","Median Interactivity (tok/s/user)","P99 Interactivity (tok/s/user)","Std Interactivity (tok/s/user)",
"Mean ITL (ms)","Median ITL (ms)","P99 ITL (ms)","Std ITL (ms)",
"Mean E2E Latency (ms)","Median E2E Latency (ms)","P99 E2E Latency (ms)","Std E2E Latency (ms)",
"Disaggregated","Num Prefill GPUs","Num Decode GPUs","Spec Decoding","EP","DP Attention","Is Multinode"]


def g(d, k, default=0.0):
    v = d.get(k, default)
    return v if v is not None else default


def intvty(tpot_ms):
    return 1000.0 / tpot_ms if tpot_ms else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--pattern", required=True, help="glob with {con}")
    ap.add_argument("--cons", required=True)
    ap.add_argument("--tp", type=int, required=True)
    ap.add_argument("--isl", type=int, required=True)
    ap.add_argument("--osl", type=int, default=1024)
    ap.add_argument("--model", default="model")
    ap.add_argument("--hardware", default="mi355x")
    ap.add_argument("--hardware-key", default=None)
    ap.add_argument("--framework", default="atom")
    ap.add_argument("--precision", default="fp8")
    ap.add_argument("--spec", default="none", help="Spec Decoding col, e.g. eagle3/mtp/none")
    ap.add_argument("--date", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cons = [int(x) for x in args.cons.split(",") if x.strip()]
    hw_key = args.hardware_key or f"{args.hardware}_{args.framework}"
    rows = []
    for c in cons:
        hits = glob.glob(os.path.join(args.results_dir, args.pattern.replace("{con}", str(c))))
        if not hits:
            print(f"  [skip] con={c}: no file"); continue
        d = json.load(open(sorted(hits)[-1]))
        total = g(d, "total_token_throughput"); out = g(d, "output_throughput")
        rows.append([
            args.model, args.isl, args.osl, args.hardware, hw_key, args.framework, args.precision, args.tp, c, args.date,
            f"{total/args.tp:.6f}", f"{out/args.tp:.6f}", f"{(total-out)/args.tp:.6f}",
            f"{g(d,'mean_ttft_ms'):.6f}", f"{g(d,'median_ttft_ms'):.6f}", f"{g(d,'p99_ttft_ms'):.6f}", f"{g(d,'std_ttft_ms'):.6f}",
            f"{g(d,'mean_tpot_ms'):.6f}", f"{g(d,'median_tpot_ms'):.6f}", f"{g(d,'p99_tpot_ms'):.6f}", f"{g(d,'std_tpot_ms'):.6f}",
            f"{intvty(g(d,'mean_tpot_ms')):.6f}", f"{intvty(g(d,'median_tpot_ms')):.6f}", f"{intvty(g(d,'p99_tpot_ms')):.6f}", "",
            f"{g(d,'mean_itl_ms'):.6f}", f"{g(d,'median_itl_ms'):.6f}", f"{g(d,'p99_itl_ms'):.6f}", f"{g(d,'std_itl_ms'):.6f}",
            f"{g(d,'mean_e2el_ms'):.6f}", f"{g(d,'median_e2el_ms'):.6f}", f"{g(d,'p99_e2el_ms'):.6f}", f"{g(d,'std_e2el_ms'):.6f}",
            "false", "", "", args.spec, 1, "false", "false",
        ])
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(HEADER); w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
