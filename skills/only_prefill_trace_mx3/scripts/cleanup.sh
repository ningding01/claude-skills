#!/usr/bin/env bash
# Stop agent-bench + the ATOM server and confirm GPU memory is freed.
# SIGTERM often does NOT stop the ATOM server; escalate to SIGKILL by pattern.
set -u
pkill -f "agentbench.cli agent" 2>/dev/null && echo "bench: SIGTERM sent" || echo "bench: none"

if pgrep -f "atom.entrypoints.openai_server" >/dev/null; then
  pkill -TERM -f "atom.entrypoints.openai_server" 2>/dev/null
  echo "server: SIGTERM sent, waiting..."
  sleep 8
fi
if pgrep -f "atom.entrypoints.openai_server" >/dev/null; then
  echo "server still up -> SIGKILL"
  pkill -9 -f "atom.entrypoints.openai_server" 2>/dev/null
  sleep 5
fi

echo "--- residual processes ---"
pgrep -af "openai_server|agentbench.cli" | grep -v pgrep || echo "none ✓"
echo "--- GPU VRAM used (idle ~300MB) ---"
rocm-smi --showmeminfo vram 2>/dev/null | grep -i "used memory"
