#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

PYTHONPATH=./src \
python -m m2asda.subtype \
  --read_path fake_m2asda_data/asc_all.h5ad \
  --pth_path fake_m2asda_data/phase1_G.pth \
  --save_path fake_m2asda_data/subtype.csv \
  --num_types 2 \
  --n_epochs 20 --batch_size 128