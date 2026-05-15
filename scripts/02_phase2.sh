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
  --out_h5ad data/corrected_tgt_ref.h5ad \
  --out_model ckpt/phase2.pth \
