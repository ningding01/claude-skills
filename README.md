# claude-skills

A collection of reusable Claude skills.

## Install
Clone individual skills into your Claude skills dir:
```bash
git clone https://github.com/ningding01/claude-skills.git
cp -r claude-skills/skills/<skill-name> ~/.claude/skills/
```
Or symlink the whole `skills/` directory.

## Skills
| Skill | Description |
|---|---|
| [deepseek-sparse-bench](skills/deepseek-sparse-bench) | Benchmark DeepSeek-V4 sparse attention (prefill & decode) on AMD aiter / gfx950; align a NVIDIA FlashMLA run on B300. |
