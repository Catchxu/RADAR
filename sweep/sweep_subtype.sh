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

# 任务标签，写进输出目录和日志名
RUN_TAG="${RUN_TAG:-10x3v2}"

# 只在“变量未定义”时给默认值；如果你显式传 TARGET_ASSAY=""，就保留空串
if [[ ! "${TARGET_ASSAY+x}" ]]; then
  TARGET_ASSAY="10x 3' v2"
fi

# seed 分片
SEED_START="${SEED_START:-0}"
NUM_RANDOM_SEEDS="${NUM_RANDOM_SEEDS:-1000}"

# sweep 超参数列表
NUM_TYPES_LIST="${NUM_TYPES_LIST:-2}"
N_EPOCHS_LIST="${N_EPOCHS_LIST:-1000}"
BATCH_SIZE_LIST="${BATCH_SIZE_LIST:-128,64}"
LEARNING_RATE_LIST="${LEARNING_RATE_LIST:-1e-4}"
WEIGHT_DECAY_LIST="${WEIGHT_DECAY_LIST:-0}"

OUTDIR="/data1011/yuzimu/M2ASDA/result/sweep_subtype_${RUN_TAG}_${BATCH_SIZE_LIST}"
GPU_TAG="${GPU//:/}"
OUTCSV="${OUTDIR}/summary_${RUN_TAG}_${GPU_TAG}_seed${SEED_START}_n${NUM_RANDOM_SEEDS}.csv"

mkdir -p "${OUTDIR}"

ARGS=(
  --read_path data/corrected_tgt_colorectum.h5ad
  --pred_csv output/phase1_pred_ASCs.csv
  --pth_path ckpt/phase1_G.pth
  --output_csv "${OUTCSV}"
  --cell_col cell_id
  --label_col pred
  --anomaly_label abnormal
  --truth_label_col cell_state_label
  --truth_subtype_values tumor02_MMRd,tumor02_MMRp
  --num_types_list "${NUM_TYPES_LIST}"
  --n_epochs_list "${N_EPOCHS_LIST}"
  --batch_size_list "${BATCH_SIZE_LIST}"
  --learning_rate_list "${LEARNING_RATE_LIST}"
  --weight_decay_list "${WEIGHT_DECAY_LIST}"
  --num_random_seeds "${NUM_RANDOM_SEEDS}"
  --seed_start "${SEED_START}"
  --GPU "${GPU}"
)

if [[ -n "${TARGET_ASSAY}" ]]; then
  ARGS+=(
    --target_assay "${TARGET_ASSAY}"
    --adata_assay_col assay
    --csv_assay_col assay
  )
fi

PYTHONPATH=./src python -m m2asda.sweep_subtype "${ARGS[@]}"