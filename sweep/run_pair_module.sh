#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHONPATH=./src \
python -u -m m2asda.train_pair_module \
  --ref_path data/ref_clean_colorectum.h5ad \
  --tgt_path data/tgt_clean_colorectum.h5ad \
  --pred_path output/phase1_pred_ASCs.csv \
  --out_dir output/pair_module \
  --device cuda:0 \
  --seed 2026 \
  --lr 1e-4 \
  --weight_decay 5e-4 \
  --batch_size 256 \
  --epochs 5000 \
  --attn_dim 256 \
  --alpha_entropy_weight 0 \
  --run_ig \
  --ig_steps 32 \
  --internal_batch_size 4 \
  --score_mode positive