#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p output/subtype_by_assay

PYTHONPATH=./src \
python -m m2asda.subtype \
  --read_path data/tgt_clean_colorectum.h5ad \
  --pth_path ckpt/phase1_G.pth \
  --pred_csv output/phase1_pred_ASCs.csv \
  --target_assay "10x 3' v2" \
  --adata_assay_col assay \
  --csv_assay_col assay \
  --cell_col cell_id \
  --label_col pred \
  --anomaly_label abnormal \
  --save_dir output/subtype_by_assay \
  --num_types 2 \
  --n_epochs 1000 \
  --batch_size 128 \
  --weight_decay 0.0 \
  --random_state 140 \
  --GPU cuda:0