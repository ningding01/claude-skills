"""Kernel -> module rule sets, keyed by MODEL FAMILY.

Each rule set is built so that the module names reflect THAT model's own call
structure (not a one-size-fits-all scheme). Patterns are lowercase substrings and
deliberately cover MULTIPLE frameworks (sglang/ATOM, vLLM, ...) and GPUs
(gfx942/gfx950 tensile `Cijk_*`, CK `ck::kernel_*`, AITER, triton) for the SAME
logical module, so cross-framework / cross-GPU comparisons land in the same rows.

To add a model or framework:
  1. copy an existing block, rename modules to the new model's structure,
  2. add kernel-name substrings (run build_xlsx.py --coverage first to see them),
  3. register it in RULESETS + resolve_ruleset().

Rule order matters: FIRST match wins. Put specific buckets before generic GEMM /
elementwise catch-alls (e.g. a fused "qk_norm_rope" before plain "rmsnorm"; MoE
"grouped_gemm" before generic "gemm").
"""

# ----------------------------------------------------------------------------- M3
MINIMAX_M3 = {
    "order": [
        "layernorm (Gemma RMSNorm)",
        "proj GEMM (q/kv/o + shared-expert)",
        "QK norm + RoPE + KV/index cache",
        "lightning indexer (index attn scores)",
        "top-k block select",
        "sparse main attention (top-k blocks)",
        "full attention (dense layers)",
        "MoE router / sort",
        "MoE expert GEMM + SwiGLU-OAI (MXFP8)",
        "TP all-reduce",
        "MXFP8 quant",
        "elementwise / copy / index / sample",
        "other/unclassified",
    ],
    "rules": [
        ("TP all-reduce", ("nccldevkernel", "mscclkernel", "cross_device_reduce", "allgather", "all_reduce")),
        ("QK norm + RoPE + KV/index cache", ("_sparse_qk_index_gemma_rmsnorm_rope", "_qk_gemma_rmsnorm_rope", "qk_norm_rope", "store_kvcache")),
        ("layernorm (Gemma RMSNorm)", ("gemma_fused_add_rmsnorm", "gemma_rmsnorm", "_rmsnorm", "add_rmsnorm")),
        ("lightning indexer (index attn scores)", ("block_score", "_decode_score", "mqa_logits", "_indexer")),
        ("top-k block select", ("_topk_index", "minimax_decode_topk_block", "radix_topk", "topk")),
        ("sparse main attention (top-k blocks)", ("_gqa_share_sparse", "_merge_topk_attn_out", "sparse_attn")),
        ("full attention (dense layers)", ("_fwd_grouped_kernel", "_fwd_kernel", "paged_decode")),
        ("MoE expert GEMM + SwiGLU-OAI (MXFP8)", ("grouped_gemm", "moe_mxgemm", "mfma_moe", "_swiglu_oai", "add_clamp_mul_sigmoid_split", "act_and_mul")),
        ("MoE router / sort", ("moe_align_block_size", "count_and_sort_expert", "_router_triton", "moe_sort", "moe_sorting")),
        ("MXFP8 quant", ("_mxfp8_quant", "scaled_quant", "per_token_group_quant", "group_quant")),
        ("proj GEMM (q/kv/o + shared-expert)", ("_mxfp8_linear", "cijk_", "hgemm", "ck::kernel_gemm", "_gemm_a", "bf16gemm", "fp8gemm")),
        ("elementwise / copy / index / sample", ("at::native", "amd_rocclr", "memset", "memcpy", "rocprim", "elementwise", "reduce_kernel",
                                                  "index_", "vectorized", "copybuffer", "catarray", "fill", "greedy_sample", "_sample", "gather", "scatter", "where", "cast", "clamp",
                                                  "write_req_to_token_pool", "create_flashinfer_kv_indices", "compute_position", "kv_splits", "triton_poi_fused")),
    ],
    # include the DENSE-layer decode attention (split-KV stage1/stage2) so the
    # prefill/decode boundary lands at the START of the decode step (the 3 dense
    # layers run before the first SPARSE decode marker; without these they leak
    # into the prefill region and decode loses its "full attention (dense layers)").
    "decode_markers": ("_gqa_share_sparse_decode", "_decode_score", "_merge_topk_attn_out",
                       "minimax_decode_topk_block", "_fwd_grouped_kernel_stage1", "_fwd_kernel_stage2"),
}

# ------------------------------------------------------------------------- DSV4
# Module scheme taken from DeepSeek-V4 ATOM/vLLM reference; covers both stacks.
DEEPSEEK_V4 = {
    "order": [
        "rms_norm (pre/post)",
        "dense_gemm (q/kv/o + shared expert)",
        "DSA: Q/K rope",
        "DSA: indexer (lightning index)",
        "DSA: compressor (CSA/HCA)",
        "DSA: topk select",
        "DSA: sparse attention core",
        "MoE: router/gate/sort",
        "MoE: expert gemm (fc1/fc2/act)",
        "comm (all-reduce/all-gather)",
        "quant (fp8/fp4)",
        "memcpy/index/other",
        "other/unclassified",
    ],
    "rules": [
        ("comm (all-reduce/all-gather)", ("nccldevkernel", "cross_device_reduce", "allgather", "all_reduce")),
        ("DSA: indexer (lightning index)", ("mqa_logits", "indexer", "topk_softplus")),
        ("DSA: compressor (CSA/HCA)", ("compress_attn", "compressor_states", "hca_compress", "compress_norm_rope_insert_sparse", "compressed_slot")),
        ("DSA: Q/K rope", ("_inverse_rope", "qnormropekvrope", "qk_norm_rope", "norm_rope_scatter")),
        ("rms_norm (pre/post)", ("add_rmsnorm", "rmsnorm2d", "_rmsnorm", "_fused_q_kv_rmsnorm", "hc_prenorm_gemm", "prenorm")),
        ("DSA: topk select", ("radix_topk", "gathertopk", "topkperrow", "topk_lens", "pack_global_topk", "build_c128a_topk", "combine_topk", "topk")),
        ("DSA: sparse attention core", ("mhc_pre", "mhc_post", "paged_decode", "sparse_attn", "pa_prefill", "_swa_write", "csa_translate", "swa_indices", "_paged_")),
        ("MoE: router/gate/sort", ("moe_sort", "moe_sorting", "moe_align", "router", "count_and_sort")),
        ("MoE: expert gemm (fc1/fc2/act)", ("moe_mxgemm", "mfma_moe", "act_and_mul", "rope_hadamard_rotate_activation", "moe1", "moe2", "grouped_gemm")),
        ("quant (fp8/fp4)", ("scaled_quant", "per_token_group_quant", "dequantize_and_gather", "_quant")),
        ("dense_gemm (q/kv/o + shared expert)", ("ck::kernel_gemm", "_gemm_a8w8", "_gemm_a16", "cijk_", "bf16gemm", "fp8gemm", "hgemm", "gemm")),
        ("memcpy/index/other", ("at::native", "catarray", "memcpy", "memset", "copybuffer", "rocprim", "elementwise", "reduce_kernel",
                                  "index_", "vectorized", "fill", "gather", "scatter", "softplus", "silu", "where", "sample", "trampoline")),
    ],
    "decode_markers": ("_paged_decode", "_sparse_attn_decode", "topkperrowdecode", "_v4_paged_decode_indices", "decode_ragged"),
}

# ----------------------------------------------------------------------- generic
# Model-agnostic fallback. Buckets are coarse but cover unknown models/frameworks.
GENERIC = {
    "order": [
        "norm",
        "attention",
        "MoE router/sort",
        "MoE expert gemm/act",
        "dense gemm/proj",
        "comm (all-reduce/all-gather)",
        "quant",
        "elementwise/copy/index",
        "other/unclassified",
    ],
    "rules": [
        ("comm (all-reduce/all-gather)", ("nccl", "msccl", "all_reduce", "allreduce", "allgather", "all_gather", "cross_device_reduce", "reduce_scatter")),
        ("norm", ("rmsnorm", "rms_norm", "layernorm", "layer_norm", "add_rmsnorm", "_norm")),
        ("MoE router/sort", ("moe_sort", "moe_sorting", "moe_align", "router", "count_and_sort", "topk_softmax")),
        ("MoE expert gemm/act", ("moe_mxgemm", "grouped_gemm", "mfma_moe", "moe1", "moe2", "expert", "swiglu", "act_and_mul", "silu_and_mul")),
        ("attention", ("attn", "attention", "flash", "fmha", "mha", "mla", "sdpa", "paged", "_fwd_kernel", "sparse", "block_score", "decode_score", "mqa_logits", "topk_index", "rope")),
        ("quant", ("quant", "scaled_quant", "dequant")),
        ("dense gemm/proj", ("gemm", "cijk_", "hgemm", "matmul", "cutlass", "linear", "ck::kernel_gemm")),
        ("elementwise/copy/index", ("at::native", "elementwise", "memcpy", "memset", "copybuffer", "reduce_kernel", "index_", "vectorized",
                                     "catarray", "fill", "gather", "scatter", "rocprim", "sample", "where", "cast")),
    ],
    "decode_markers": ("decode", "_paged_decode", "decode_attn"),
}

RULESETS = {
    "minimax-m3": MINIMAX_M3,
    "deepseek-v4": DEEPSEEK_V4,
    "generic": GENERIC,
}


def resolve_ruleset(model: str):
    """Map a free-form model/ruleset label to a registered rule set."""
    if not model:
        return "generic", GENERIC
    m = model.lower().replace("_", "-")
    if m in RULESETS:
        return m, RULESETS[m]
    if "m3" in m or "minimax" in m:
        return "minimax-m3", MINIMAX_M3
    if "deepseek" in m or "dsv4" in m or "v4" in m:
        return "deepseek-v4", DEEPSEEK_V4
    return "generic", GENERIC
