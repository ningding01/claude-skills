# Extending the rule sets (new model / framework / GPU)

The whole point of this skill: modules follow **the model's own call structure**, so
adding a model means writing a small rule set in `scripts/rulesets.py`.

## Workflow for a NEW model or framework

1. Write a spec that points at the trace(s) and set `"model"` to the closest existing
   key (or a new name). Run coverage first:

   ```
   python scripts/build_xlsx.py myspec.json --coverage
   ```

   It prints, per dataset, the ruleset used, total ms, and **unclassified %** plus the
   **top unclassified kernels**. If unclassified > ~3%, the ruleset doesn't fit.

2. Open `scripts/rulesets.py`, copy the closest block, rename the modules to match how
   THIS model is built (e.g. dense model: just `attention`, `mlp gemm`, `norm`; MLA model:
   `q down/up`, `kv compress`, `MLA core`; MoE: split router vs expert gemm). Add the
   unclassified kernel-name substrings to the right module.

3. Register it in `RULESETS` and add a branch in `resolve_ruleset()` so the `"model"`
   label maps to it.

4. Re-run `--coverage` until unclassified is small, then run without `--coverage`.

## Rule mechanics

- Patterns are **lowercase substrings**; FIRST match wins, so ordering matters.
- Put SPECIFIC buckets before generic catch-alls:
  fused `qk_norm_rope` before plain `rmsnorm`; MoE `grouped_gemm` before generic `gemm`;
  `block_score`/`decode_score` (indexer) before generic `attn`/`_fwd_kernel`.
- Cover multiple frameworks in the SAME module so cross-framework comparisons align:
  e.g. MoE expert gemm = `("_mxfp8_grouped_gemm", "moe_mxgemm", "mfma_moe", ...)` catches
  sglang, CK, and AITER names. Same for GPUs: tensile `cijk_`, CK `ck::kernel_gemm`, `hgemm`.

## Phase split (prefill vs decode)

**Prefer profile_by_stage.** If you have stage-tagged files (`*-EXTEND*` = prefill,
no-suffix = decode), give them as `stage_trace` and the skill uses them directly
(`region:"all"`, exact boundary, no marker guessing). Only fall back to slicing a
`combined_trace` (`phase` + `decode_markers`) when stage files don't exist.

`region:"prefill"|"decode"` splits a COMBINED trace at the first timestamp of a
**decode-marker** kernel (`decode_markers` in the ruleset). For a new model, set
`decode_markers` to kernels that ONLY fire during decode (`*_decode_*`, `paged_decode`,
`topKPerRowDecode`, …).

**Gotcha — include DENSE-layer decode attention in `decode_markers`.** In hybrid models
(e.g. M3: 3 dense + 57 sparse layers) the dense layers run FIRST in each decode step,
before any sparse decode kernel. If `decode_markers` lists only sparse decode kernels,
the boundary lands *after* the dense layers, so they leak into the prefill region and the
decode breakdown shows no full attention. Fix: add the dense-layer decode attention
kernels (M3: `_fwd_grouped_kernel_stage1`, `_fwd_kernel_stage2`). Verify they are
decode-only (small total count, all at the tail). If markers aren't found at all, the
whole trace is returned with a warning.

## CUDA-graph decode caveat (important)

When decode runs under a CUDA graph, the kineto trace records the graphed kernels
~**once** regardless of how many steps ran. So a decode region's GPU-busy total ≈ ONE
step, which matches the measured wall-clock ms/step. **Do not divide the decode region
by the number of generated tokens.** For absolute per-step latency, also measure by
timing `(t[N+1 tok] - t[1 tok]) / N` and sanity-check against the trace.

## Comparison dimensions supported

Each sheet's `compare` list can mix datasets that differ by GPU, framework, or both
(same model assumed for aligned modules). Ratio columns are computed vs the FIRST
dataset. Examples:
- same model + same framework + different GPU  (MI355 vs MI300X)
- same model + different framework + same GPU  (sglang vs vLLM)
- same model + different framework + different GPU
Cross-MODEL sheets also work (module rows are the union; non-shared modules show n/a).
