#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

PYTHONPATH=./src \
python -m m2asda.correct \
  --read_path fake_m2asda_data/ref.h5ad fake_m2asda_data/tgt1.h5ad fake_m2asda_data/tgt2.h5ad \
  --pth_path fake_m2asda_data/phase1_G.pth \
  --save_path fake_m2asda_data/corrected.h5ad \
  --phase1_csv_paths fake_m2asda_data/phase1_tgt1.csv fake_m2asda_data/phase1_tgt2.csv \
  --n_epochs_p 2 --n_critic_p 1 \
  --n_epochs_c 2 --n_critic_c 1 \
  --batch_size_c 128