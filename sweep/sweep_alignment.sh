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

OUTDIR="/data1011/yuzimu/M2ASDA/result/sweep_phase2_alignment_gan_tgtbatch"
GPU_TAG="${GPU//:/}"
OUTCSV="${OUTDIR}/summary_${GPU_TAG}.csv"

mkdir -p "${OUTDIR}"

SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-9}"
SEEDS="$(seq -s, "${SEED_START}" "${SEED_END}")"

PAIR_CSV="${PAIR_CSV:-/data1011/yuzimu/M2ASDA/output/pair_module/target_ref_pairs_ig.csv}"
PHASE1_CKPT="${PHASE1_CKPT:-/data1011/yuzimu/M2ASDA/ckpt/phase1_G.pth}"

echo "GPU: ${GPU}"
echo "OUTDIR: ${OUTDIR}"
echo "OUTCSV: ${OUTCSV}"
echo "PAIR_CSV: ${PAIR_CSV}"
echo "PHASE1_CKPT: ${PHASE1_CKPT}"
echo "Seeds: ${SEED_START}-${SEED_END}"

PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" python -u -m m2asda.sweep_alignment \
  --ref_h5ad data/ref_clean_colorectum.h5ad \
  --target_h5ad data/tgt_clean_colorectum.h5ad \
  --pair_csv "${PAIR_CSV}" \
  --phase1_ckpt "${PHASE1_CKPT}" \
  --outdir "${OUTDIR}" \
  --output_csv "${OUTCSV}" \
  --batch_key assay \
  --ref_bio_key cell_type \
  --tgt_bio_key cell_state_label \
  --batch_size 512 \
  --num_workers 0 \
  --seed_list "${SEEDS}" \
  --epochs_list 50,100 \
  --hidden_dims 128,256 \
  --g_blocks_list 6 \
  --d_blocks_list 3 \
  --cond_dim 128 \
  --g_expansion 4 \
  --g_dropouts 0 \
  --d_dropouts 0.2 \
  --delta_scales 0.5,1.0 \
  --n_candidates_list 16,32 \
  --use_attention_ref_list 0 \
  --loss_presets no_tgt_batch,batch_conf,batch_conf_mid,batch_conf_strong \
  --phase1_hidden_dim 512 \
  --phase1_latent_dim 128 \
  --phase1_memory_size 512 \
  --phase1_num_heads 8 \
  --phase1_temperature 1.0 \
  --phase1_dropout 0 \
  --phase1_use_memory_bank \
  --lr_g_list 1e-4 \
  --lr_d_list 1e-4 \
  --weight_decay_list 1e-4 \
  --grad_clip 2.0 \
  --save_every 999999 \
  --log_every 50 \
  --k_metric 30 \
  --batchkl_sample_n 100 \
  --leiden_res 1.0 \
  --device "${GPU}" \
  --skip_existing \
  "$@"