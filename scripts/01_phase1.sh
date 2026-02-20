#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

PYTHONPATH=./src \
python -m m2asda.anomaly \
  --ref_path fake_m2asda_data/ref.h5ad \
  --tgt_path fake_m2asda_data/tgt1.h5ad \
  --result_path fake_m2asda_data/phase1_pred_tgt1.csv \
  --pth_path fake_m2asda_data/phase1_G.pth \
  --n_epochs 10 --batch_size 256 --n_critic 2