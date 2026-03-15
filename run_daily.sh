#!/usr/bin/env bash
set -euo pipefail
cd /home/clawuser/.openclaw/workspace/holo-repro
. .venv/bin/activate
python src/experiment.py --mode both --n 40 --seed 42
