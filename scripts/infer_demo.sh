#!/usr/bin/env bash

cd "$(dirname "$0")/.."

export PYTHONPATH="$PWD/src:$PYTHONPATH"

python -u -m radar.infer_pipeline \
  --ref_path data/ref_clean_colorectum.h5ad \
  --tgt_path data/tgt_clean_colorectum.h5ad \
  --phase1_ckpt ckpt/phase1_G.pth \
  --phase2_ckpt ckpt/phase2.pth \
  --phase3_ckpt ckpt/phase3.pth \
  --pred_csv output/phase1_pred_ASCs.csv \
  --out_dir result \
  --device cuda:0