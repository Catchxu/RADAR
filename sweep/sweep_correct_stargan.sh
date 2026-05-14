#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

GPU="${GPU:-cuda:0}"
OUTDIR="/data1011/yuzimu/M2ASDA/result/sweep_phase2_traditional_loss"

GPU_TAG="${GPU//:/}"
OUTCSV="${OUTDIR}/summary_${GPU_TAG}.csv"

mkdir -p "${OUTDIR}"

SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-9}"
SEEDS="$(seq -s, "${SEED_START}" "${SEED_END}")"

echo "GPU: ${GPU}"
echo "OUTDIR: ${OUTDIR}"
echo "Seeds: ${SEED_START}-${SEED_END}"

PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" python -u -m m2asda.sweep_correct_stargan \
  --ref_h5ad data/ref_clean_colorectum.h5ad \
  --tgt_h5ad data/tgt_clean_colorectum.h5ad \
  --outdir "${OUTDIR}" \
  --output_csv "${OUTCSV}" \
  --batch_key assay \
  --disease_key disease \
  --normal_value normal \
  --ref_bio_key cell_type \
  --bio_key cell_state_label \
  --batch_size 512 \
  --align_batch_size 1024 \
  --d_steps_list 1 \
  --g_steps_list 1 \
  --domain_balance_power_list 0.5 \
  --num_workers 0 \
  --seed_list "${SEEDS}" \
  --epochs_list 500 \
  --hidden_dims 512 \
  --cond_dims 128 \
  --g_num_blocks_list 6 \
  --d_num_blocks_list 3 \
  --g_dropouts 0 \
  --d_dropouts 0.2 \
  --use_change_gate_list 0 \
  --loss_presets With_state_supervision \
  --device "${GPU}" \
  --show_progress \
  --save_passed_h5ad \
  "$@"