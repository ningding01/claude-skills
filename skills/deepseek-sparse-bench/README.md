# deepseek-sparse-bench (Claude skill)

Benchmark DeepSeek-V4 sparse attention (prefill & decode) on AMD aiter / gfx950.

## Install as a Claude skill
```bash
git clone <this-repo-url> ~/.claude/skills/deepseek-sparse-bench
```
Then in Claude Code the skill `deepseek-sparse-bench` is available; see SKILL.md for usage.

## Quick run
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True python scripts/prefill_sparse_bench.py --num-tokens 262144 --heads 128 --topk 1024 --dtype bf16
PYTORCH_ALLOC_CONF=expandable_segments:True python scripts/decode_sparse_bench.py  --batch 256 --context 65536 --topk 1024 --dtype fp8
```
