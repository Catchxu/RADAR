#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."



PYTHONPATH=./src \
python -m radar.alignment \
  --ref_h5ad data/ref_clean_colorectum.h5ad \
  --tgt_h5ad data/tgt_clean_colorectum.h5ad \
  --batch_key assay \
  --disease_key disease \
  --normal_value normal \
  --ref_bio_key cell_type \
  --tgt_bio_key cell_state_label \
  --out_h5ad data/corrected_tgt_ref.h5ad \
  --out_model ckpt/phase2.pth \
  --target_domain "10x 3' v2" \
  --align_batch_size 1024 \
  --epochs 500 \
  --batch_size 512 \
  --d_steps 1 \
  --g_steps 1 \
  --domain_balance_power 0.5 \
  --lr_g 2.0e-4 \
  --lr_d 2.0e-4 \
  --hidden_dim 256 \
  --cond_dim 128 \
  --g_num_blocks 6 \
  --d_num_blocks 3 \
  --g_dropout 0.0 \
  --d_dropout 0.1 \
  --lambda_batch 3.0 \
  --lambda_state 0.2 \
  --lambda_rec 2.0 \
  --lambda_id 0.4 \
  --device cuda:0 \
  --seed 42