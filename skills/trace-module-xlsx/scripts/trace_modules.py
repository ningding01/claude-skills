"""Engine: load a kineto/perfetto trace (.json or .json.gz), optionally split it
into prefill/decode regions, classify GPU kernels into model-native modules, and
aggregate GPU-busy time per module and per kernel.

Stdlib only. Used by build_xlsx.py.
"""
import collections
import glob
import gzip
import json
import re

from rulesets import resolve_ruleset


def load_kernels(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        d = json.load(f)
    ev = d.get("traceEvents", d if isinstance(d, list) else [])
    return [e for e in ev if e.get("cat") == "kernel" and "dur" in e and "ts" in e]


def _sibling_rank_files(path):
    """Given a path containing TP-<n>, return all sibling TP-* files."""
    pat = re.sub(r"TP-\d+", "TP-*", path)
    files = sorted(glob.glob(pat))
    return files or [path]


def classify(name, rules):
    n = name.lower()
    for mod, keys in rules:
        if any(k in n for k in keys):
            return mod
    return "other/unclassified"


def split_region(events, markers, region):
    """region in {all, prefill, decode}. Uses decode-marker kernels to find the
    prefill/decode boundary in a COMBINED trace. Returns (events, note)."""
    if region == "all":
        return events, ""
    mts = [e["ts"] for e in events if any(k in e["name"].lower() for k in markers)]
    if not mts:
        return events, f"WARN: no decode markers found; '{region}' returned whole trace"
    start = min(mts)
    if region == "prefill":
        return [e for e in events if e["ts"] < start], ""
    if region == "decode":
        return [e for e in events if e["ts"] >= start], ""
    raise ValueError(f"bad region {region}")


def analyze(trace_glob, model, region="all", avg_ranks=False):
    """Return dict: {ruleset, total_ms, modules:{mod:ms}, kernels:{mod:{name:[calls,ms]}}, note}."""
    rs_name, rs = resolve_ruleset(model)
    files = sorted(glob.glob(trace_glob))
    if not files:
        raise FileNotFoundError(trace_glob)
    base = files[0]
    rank_files = _sibling_rank_files(base) if avg_ranks else [base]

    mod_sum = collections.defaultdict(float)   # averaged ms across ranks
    note = ""
    n = len(rank_files)
    for rf in rank_files:
        ev = load_kernels(rf)
        ev, note0 = split_region(ev, rs["decode_markers"], region)
        note = note or note0
        for e in ev:
            mod_sum[classify(e["name"], rs["rules"])] += e["dur"] / 1000.0 / n

    # per-kernel detail from the first rank only
    ev0 = load_kernels(base)
    ev0, _ = split_region(ev0, rs["decode_markers"], region)
    kdet = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0.0]))
    for e in ev0:
        m = classify(e["name"], rs["rules"])
        kdet[m][e["name"]][0] += 1
        kdet[m][e["name"]][1] += e["dur"] / 1000.0

    total = sum(mod_sum.values())
    return {
        "ruleset": rs_name,
        "order": rs["order"],
        "total_ms": total,
        "modules": dict(mod_sum),
        "kernels": kdet,
        "ranks": n,
        "note": note,
    }


def merged_order(orders):
    """Union of several module orders, preserving first-seen order; unclassified last."""
    out = []
    for o in orders:
        for m in o:
            if m not in out:
                out.append(m)
    if "other/unclassified" in out:
        out.remove("other/unclassified")
        out.append("other/unclassified")
    return out
