# holo-repro

CPU-only daily reproduction workflow for arXiv:2601.21065 (Building Holographic Entanglement by Measurement).

## What this does
- Builds small graph-based toy instances (disk/wormhole-like templates)
- Computes a Gaussian-state-inspired boundary entropy proxy
- Compares against a minimal-surface proxy via graph min-cut
- Outputs daily markdown + JSON metrics

## Run once
```bash
cd /home/clawuser/.openclaw/workspace/holo-repro
python3 src/experiment.py --mode both --n 32 --seed 42
```

## Daily report script
```bash
bash run_daily.sh
```
