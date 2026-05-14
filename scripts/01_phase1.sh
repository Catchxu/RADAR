#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

PYTHONPATH=./src \
python -u -m radar.detection \
  --ref_path data/ref_clean_colorectum.h5ad \
  --tgt_path data/tgt_clean_colorectum.h5ad \
  --result_path output/phase1_pred_tgt.csv \
  --pth_path ckpt/phase1_G.pth \
  --GPU cuda:0 \
  --batch_size 512 \
  --random_state 2026 \
  --normalization 0 \
  --n_epochs 300