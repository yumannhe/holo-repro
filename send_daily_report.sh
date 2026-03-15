#!/usr/bin/env bash
set -euo pipefail

cd /home/clawuser/.openclaw/workspace/holo-repro

# run experiment and capture markdown output
REPORT_TEXT=$(bash run_daily.sh)

# deliver to current DM channel
openclaw agent --agent main --deliver \
  --reply-channel discord \
  --reply-to 1477003177625649314 \
  --message "$REPORT_TEXT" >/dev/null

echo "sent daily holo report"
