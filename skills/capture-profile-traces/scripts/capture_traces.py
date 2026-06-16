#!/usr/bin/env python3
"""Capture profiler traces from a RUNNING SGLang or vLLM server, in three forms:

  1. combined        : one request, prefill + N decode steps, profiled together.
  2. prefill by_stage: the prefill stage on its own.
  3. decode  by_stage: the decode stage on its own.

SGLang has profile_by_stage -> #2 and #3 are EXACT (one capture writes both an
`*-EXTEND*` prefill file and a no-suffix decode file). vLLM has NO stage tagging, so
#2/#3 are approximated by request shaping (prefill = max_tokens=1; decode = warm the
prefix cache then profile) and each capture is moved into its own sub-dir. For vLLM you
can also just use #1 and let the analysis skill slice it.

Exact token control: we send raw token ids (SGLang `input_ids`, vLLM `prompt=[ids]`) so
the prompt length is exactly --input-len regardless of tokenizer.

Server must already be running with the profiler dir env set at launch:
  SGLang: SGLANG_TORCH_PROFILER_DIR=<out-dir>
  vLLM:   VLLM_TORCH_PROFILER_DIR=<out-dir>
(use launch_server.sh). Stdlib only.
"""
import argparse
import json
import os
import time
import urllib.request

def _req(method, url, body=None, timeout=2400):
    data = None
    if body is not None:
        data = json.dumps(body).encode()
    r = urllib.request.Request(url, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return resp.status, resp.read().decode()

def get(url, timeout=10):
    try:
        return _req("GET", url, None, timeout)
    except Exception as e:
        return None, str(e)

def post(url, body=None, timeout=2400):
    return _req("POST", url, body if body is not None else {}, timeout)


def wait_health(base, secs=1800):
    print(f"[wait] {base}/health ...")
    t0 = time.time()
    while time.time() - t0 < secs:
        st, _ = get(base + "/health", timeout=5)
        if st == 200:
            print(f"[wait] ready after {int(time.time()-t0)}s")
            return True
        time.sleep(10)
    raise TimeoutError("server not healthy in time")


def build_ids(n, base, mod):
    return [(i % mod) + base for i in range(n)]


def snapshot(d):
    try:
        return {f: os.path.getmtime(os.path.join(d, f)) for f in os.listdir(d)}
    except FileNotFoundError:
        return {}


def new_files(d, before, wait=180):
    """Poll until trace files newer than `before` appear and stop growing; return list."""
    pat = (".json", ".json.gz", ".trace.json", ".pt.trace.json")
    t0 = time.time()
    last = -1
    while time.time() - t0 < wait:
        cur = snapshot(d)
        new = [f for f in cur if (f not in before or cur[f] > before.get(f, 0))
               and f.endswith(pat)]
        if new and len(new) == last:
            return sorted(new)
        last = len(new)
        time.sleep(3)
    return sorted([f for f in snapshot(d) if f not in before and f.endswith(pat)])


# ---------------------------------------------------------------- SGLang
def sg_generate(base, ids, k):
    post(base + "/generate", {
        "input_ids": ids,
        "sampling_params": {"max_new_tokens": k, "temperature": 0.0, "ignore_eos": True},
        "stream": False,
    })


def sg_profile(base, prefix, out, by_stage, num_steps):
    body = {"output_dir": out, "activities": ["CPU", "GPU"],
            "with_stack": False, "record_shapes": False, "profile_prefix": prefix}
    if by_stage:
        body["profile_by_stage"] = True
        body["num_steps"] = num_steps
    post(base + "/start_profile", body)


def capture_sglang(a, base, ids):
    out = a.out_dir
    # 1) combined
    print("[sglang] combined ...")
    sg_profile(base, f"{a.prefix}_combined", out, by_stage=False, num_steps=0)
    sg_generate(base, ids, a.combined_decode)
    post(base + "/stop_profile")
    time.sleep(8)
    # 2+3) by_stage: flush -> full prefill in EXTEND + decode (few steps)
    print("[sglang] by_stage (prefill EXTEND + decode) ...")
    post(base + "/flush_cache")
    chunks = (a.input_len + a.chunk - 1) // a.chunk
    nsteps = max(a.decode_steps, chunks + 4)
    sg_profile(base, f"{a.prefix}_stage", out, by_stage=True, num_steps=nsteps)
    sg_generate(base, ids, a.decode_steps)
    post(base + "/stop_profile")
    time.sleep(8)
    print("\n[sglang] outputs in", out)
    print(f"  combined : {a.prefix}_combined-*-TP-*.trace.json.gz")
    print(f"  prefill  : {a.prefix}_stage-*-TP-*-EXTEND.trace.json.gz  (by_stage)")
    print(f"  decode   : {a.prefix}_stage-*-TP-*.trace.json.gz         (by_stage, no suffix)")


# ---------------------------------------------------------------- vLLM
def vllm_model(base):
    st, body = get(base + "/v1/models")
    return json.loads(body)["data"][0]["id"]


def vllm_generate(base, model, ids, k):
    post(base + "/v1/completions", {
        "model": model, "prompt": ids, "max_tokens": k,
        "temperature": 0.0, "ignore_eos": True, "stream": False,
    })


def _vllm_one(base, model, ids, k, out, label):
    before = snapshot(out)
    post(base + "/start_profile")
    vllm_generate(base, model, ids, k)
    post(base + "/stop_profile")
    files = new_files(out, before)
    dst = os.path.join(out, f"{label}")
    os.makedirs(dst, exist_ok=True)
    for f in files:
        os.replace(os.path.join(out, f), os.path.join(dst, f))
    print(f"[vllm] {label}: {len(files)} file(s) -> {dst}")


def capture_vllm(a, base, ids):
    out = a.out_dir
    model = a.model or vllm_model(base)
    print("[vllm] served model:", model)
    print("[vllm] NOTE: no profile_by_stage; prefill/decode are request-shaped approximations.")
    # 1) combined
    _vllm_one(base, model, ids, a.combined_decode, out, f"{a.prefix}_combined")
    # 2) prefill-ish: max_tokens=1 (prefill + 1 decode)
    try:
        post(base + "/reset_prefix_cache")
    except Exception:
        pass
    _vllm_one(base, model, ids, 1, out, f"{a.prefix}_prefill")
    # 3) decode-ish: warm the prefix cache, then profile (prefill is a cache hit)
    vllm_generate(base, model, ids, 1)            # warm (not profiled)
    _vllm_one(base, model, ids, a.decode_steps, out, f"{a.prefix}_decode")
    print("\n[vllm] outputs under", out, "in sub-dirs *_combined / *_prefill / *_decode")
    print("       (prefill/decode are approximate; analysis skill can also slice *_combined)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--framework", choices=["sglang", "vllm"], required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--out-dir", required=True, help="= the server's *_TORCH_PROFILER_DIR")
    p.add_argument("--prefix", default="run")
    p.add_argument("--input-len", type=int, default=4000)
    p.add_argument("--decode-steps", type=int, default=8, help="decode steps for the by_stage/decode capture (keep small)")
    p.add_argument("--combined-decode", type=int, default=32, help="decode steps in the combined capture")
    p.add_argument("--chunk", type=int, default=8192, help="chunked-prefill size (for num_steps)")
    p.add_argument("--id-base", type=int, default=100)
    p.add_argument("--id-mod", type=int, default=20000, help="token ids = (i %% id_mod)+id_base; lower if model vocab is tiny")
    p.add_argument("--model", default=None, help="vLLM served model name (auto-detected if omitted)")
    a = p.parse_args()

    base = f"http://{a.host}:{a.port}"
    os.makedirs(a.out_dir, exist_ok=True)
    wait_health(base)
    ids = build_ids(a.input_len, a.id_base, a.id_mod)
    print(f"[capture] framework={a.framework} input_len={len(ids)} "
          f"decode_steps={a.decode_steps} combined_decode={a.combined_decode}")
    if a.framework == "sglang":
        capture_sglang(a, base, ids)
    else:
        capture_vllm(a, base, ids)
    print("[capture] done.")


if __name__ == "__main__":
    main()
