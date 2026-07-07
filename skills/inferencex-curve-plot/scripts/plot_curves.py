#!/usr/bin/env python3
"""Throughput/GPU vs Interactivity & vs E2E-latency curves, optionally overlaying an
official InferenceX CSV baseline.

Input "this run": a directory of InferenceX `benchmark_serving.py` result JSONs, one per
concurrency (filename contains the concurrency, matched via --pattern with `{con}`).
Baseline (optional): an official InferenceX CSV (standard schema).

Two figures are written: `<out>/cmp_interactivity.png` and `<out>/cmp_e2e.png`.
Y = Token Throughput per GPU (tok/s); X = median Interactivity (tok/s/user) or median E2E (s).
Each point is labelled with its concurrency.

KEY DEFINITIONS (see references/inferencex-csv-schema.md):
- Throughput/GPU = total_token_throughput / TP.
- Interactivity  = 1000 / median_TPOT_ms.
- E2E is recomputed for BOTH curves as `TTFT + (OSL-1)*TPOT` so the two sides use the
  same aggregation. Do NOT plot benchmark_serving's `median_e2el` against the official
  CSV `Median E2E` column directly — their median aggregations differ and inflate the gap.
- Official CSV TTFT/TPOT/E2E columns are numerically in SECONDS despite the "(ms)" header.
"""
import argparse, csv, glob, json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_run(results_dir, pattern, cons, tp, osl):
    tput, intv, e2e, conc = [], [], [], []
    for c in cons:
        hits = glob.glob(os.path.join(results_dir, pattern.replace("{con}", str(c))))
        if not hits:
            continue
        d = json.load(open(sorted(hits)[-1]))
        total = d.get("total_token_throughput") or 0.0
        mtpot = d.get("median_tpot_ms") or 0.0
        mttft = d.get("median_ttft_ms") or 0.0
        tput.append(total / tp)
        intv.append(1000.0 / mtpot if mtpot else 0.0)
        e2e.append((mttft + (osl - 1) * mtpot) / 1000.0)   # unified formula, -> seconds
        conc.append(c)
    return tput, intv, e2e, conc


def load_official(csv_path, precision, isl, osl):
    rows = [r for r in csv.DictReader((l for l in open(csv_path) if not l.startswith("#")))
            if (precision is None or r.get("Precision") == precision)
            and (isl is None or r.get("ISL") == str(isl))]
    rows.sort(key=lambda r: int(float(r["Concurrency"])))
    tput = [float(r["Throughput/GPU (tok/s)"]) for r in rows]
    intv = [float(r["Median Interactivity (tok/s/user)"]) for r in rows]
    # official TTFT/TPOT columns are in SECONDS; recompute E2E with the same formula
    e2e = [float(r["Median TTFT (ms)"]) + (osl - 1) * float(r["Median TPOT (ms)"]) for r in rows]
    conc = [int(float(r["Concurrency"])) for r in rows]
    return tput, intv, e2e, conc


def draw(ax, xkind, run, official, run_label, off_label, title_suffix):
    mt, mi, me, mc = run
    if xkind == "interactivity":
        mx, xlabel, kname = mi, "Interactivity (tok/s/user) [median]", "Interactivity"
    else:
        mx, xlabel, kname = me, "End-to-end Latency (s) [median]", "End-to-end Latency"
    if official:
        ot, oi, oe, oc = official
        ox = oi if xkind == "interactivity" else oe
        ax.plot(ox, ot, marker="s", color="#d62728", lw=2, ms=7, label=off_label)
        for x, y, c in zip(ox, ot, oc):
            ax.annotate(str(c), (x, y), fontsize=6, color="#d62728", xytext=(3, -9), textcoords="offset points")
    ax.plot(mx, mt, marker="o", color="#1f77b4", lw=2, ms=7, label=run_label)
    for x, y, c in zip(mx, mt, mc):
        ax.annotate(str(c), (x, y), fontsize=6, color="#1f77b4", xytext=(3, 4), textcoords="offset points")
    ax.set_xlabel(xlabel); ax.set_ylabel("Token Throughput per GPU (tok/s)")
    ax.set_title(f"Throughput/GPU vs. {kname}" + (f"  ·  {title_suffix}" if title_suffix else ""))
    ax.grid(True, alpha=0.3); ax.legend()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True, help="dir of benchmark_serving result JSONs")
    ap.add_argument("--pattern", required=True, help="filename glob with {con}, e.g. 'm3_*_con{con}.json'")
    ap.add_argument("--cons", required=True, help="comma list, e.g. 1,2,4,8,16,32,64,128,256")
    ap.add_argument("--tp", type=int, required=True, help="tensor-parallel size of this run")
    ap.add_argument("--osl", type=int, default=1024, help="output seq len (for E2E formula)")
    ap.add_argument("--out-dir", default=".", help="where to write the two PNGs")
    ap.add_argument("--run-label", default="this run")
    ap.add_argument("--title-suffix", default="", help="e.g. 'MiniMax-M3 · MI355X · ATOM · ISL=8k OSL=1k'")
    # official baseline (optional)
    ap.add_argument("--official-csv", default=None, help="official InferenceX CSV (standard schema)")
    ap.add_argument("--official-precision", default=None, help="filter Precision col, e.g. fp8")
    ap.add_argument("--official-isl", default=None, help="filter ISL col, e.g. 8192")
    ap.add_argument("--official-label", default="InferenceX official")
    args = ap.parse_args()

    cons = [int(x) for x in args.cons.split(",") if x.strip()]
    run = load_run(args.results_dir, args.pattern, cons, args.tp, args.osl)
    if not run[0]:
        raise SystemExit(f"no result JSONs matched {args.pattern} in {args.results_dir}")
    official = None
    if args.official_csv:
        official = load_official(args.official_csv, args.official_precision, args.official_isl, args.osl)
    os.makedirs(args.out_dir, exist_ok=True)
    for xkind, fname in (("interactivity", "cmp_interactivity.png"), ("e2e", "cmp_e2e.png")):
        fig, ax = plt.subplots(figsize=(8, 6))
        draw(ax, xkind, run, official, args.run_label, args.official_label, args.title_suffix)
        out = os.path.join(args.out_dir, fname)
        plt.tight_layout(); plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
        print("wrote", out)


if __name__ == "__main__":
    main()
