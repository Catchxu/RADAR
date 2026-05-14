#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

PYTHONPATH="${PROJECT_ROOT}/src" python -u -m m2asda.sweep_anomaly \
  --ref_path data/ref_clean_colorectum.h5ad \
  --tgt_path data/tgt_clean_colorectum.h5ad \
  --outdir result/sweep_phase1 \
  --n_epochs_list 200,300,500,700 \
  --memory_size_list 512 \
  --batch_size_list 512,256 \
  --learning_rate_list 1e-4 \
  --n_critic_list 1 \
  --gamma_list 0.1 \
  --dropout_list 0.2,0.3 \
  --normalization_list 0 \
  --use_memory_bank_list 1 \
  --random_state_list 2026,42,823,1234 \
  --GPU cuda:0 \
  "$@"