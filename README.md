# RADAR: Reference-guided Anomalous-cell Detection, Alignment, and Resolution

RADAR is a generative adversarial framework for marker-free fine-grained discovery of anomalous cells (ACs) in multi-sample and multimodal single-cell omics data.

RADAR is designed for **Cross-sample Fine-grained Anomalous-cell Discovery (CFAD)**, where anomalous cells are defined as cell populations or cell states that are present in affected tissues but absent from healthy reference tissues. Given a healthy reference dataset and one or more target datasets, RADAR performs three coupled tasks in a unified workflow:

1. **AC detection**: identify anomalous cells in target datasets relative to a healthy reference;
2. **Multi-sample alignment**: reduce cross-sample or cross-modal technical variation while preserving AC-associated biological heterogeneity;
3. **AC resolution**: resolve detected ACs into biologically coherent subtypes.

<br/>
<div align="center">
<img src="docs/images/framework.png" width="75%">
</div>
<br/>

## Overview

RADAR contains three collaborative phases:

### Phase I: Anomalous-cell detection

RADAR first trains a reconstruction-GAN on the healthy reference dataset only. Since the model learns to reconstruct normal cellular states, target cells with larger-than-expected reconstruction deviations are identified as putative anomalous cells.

### Phase II: Multi-sample alignment

RADAR excludes Phase-I-detected ACs from alignment training to reduce the risk of treating AC-associated biological variation as batch effects. For predicted normal target cells, RADAR uses an Integrated-Gradients-guided pairing module to identify biologically relevant "kin" reference cells. These kin-cell pairs are then used to train a transferring-GAN that maps target datasets into a common reference space.

### Phase III: Anomalous-cell resolution

Detected ACs are aligned into the reference space and then resolved into fine-grained AC subtypes. RADAR combines post-alignment cellular embeddings with reconstruction-deviation signals, enabling more accurate resolution of biologically distinct AC populations across samples and modalities.

## Dependencies

RADAR is implemented in Python and has been tested with Python 3.9.

Main dependencies include:

```text
anndata>=0.10.7
numpy>=1.22.4
pandas>=1.5.1
scanpy>=1.10.1
scikit-learn>=1.2.0
scipy>=1.11.4
torch>=2.0.0
tqdm>=4.64.1
captum
matplotlib
```

## Installation

RADAR is developed as a Python package. You will need to install Python, and the recommended version is Python 3.9.

You can download the package from GitHub and install it locally:

```bash
git clone https://github.com/Catchxu/RADAR.git
cd RADAR/
pip install .
```

## Quick start

After installation, RADAR can be run through three consecutive phases: anomalous-cell detection, multi-sample alignment, and anomalous-cell resolution.

### Phase I: anomalous-cell detection

Train the reference-guided reconstruction model on the healthy reference dataset and predict anomalous cells in the target dataset:

```bash
bash scripts/01_phase1.sh
```

This step generates anomalous-cell prediction results, including cell-level anomaly labels and reconstruction-deviation scores.

### Phase II: multi-sample alignment

Use predicted normal target cells to perform reference-guided multi-sample alignment:

```bash
bash scripts/02_phase2.sh
```

This step identifies kin reference cells for predicted normal target cells and maps target datasets into the reference-aligned space.

### Phase III: anomalous-cell resolution

Resolve detected anomalous cells into fine-grained AC subtypes:

```bash
bash scripts/03_phase3.sh
```

This step generates AC subtype labels and downstream results for anomalous-cell resolution.

## Data availability

All experimental datasets used in this project are available from their respective original sources. Due to file size and data-usage considerations, raw datasets are not included in this repository.

### 10x scRNA-seq datasets

- Healthy human intestinal epithelial tissue dataset (**10xG-hInt-N**) is available at GEO: [GSE185224](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE185224).
- Human colorectal cancer tissue datasets (**10xG-hCRC-T-A** and **10xG-hCRC-T-B**) are available at GEO: [GSE178341](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE178341).
- Mouse embryo dataset (**10xG-mEmb**) is available at GEO: [GSE186069](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE186069).
- Healthy human peripheral blood mononuclear cell dataset (**10xG-hHPBMC**), used as a cross-modality reference for AC detection, is available from [10x Genomics](https://www.10xgenomics.com/datasets).
- PBMC datasets for trajectory inference (**10xG-hPBMC-A**, **10xG-hPBMC-B**, and **10xG-hPBMC-C**) are available at GEO: [GSE146974](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE146974).
- Systemic lupus erythematosus PBMC datasets for AC alignment (**10xG-hPBMCSLE-A**, **10xG-hPBMCSLE-B**, and **10xG-hPBMCSLE-C**) are available at GEO: [GSE96583](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583).
- Human skin tissue datasets (**10xG-hSCC-N1** to **10xG-hSCC-N5** and **10xG-hSCC-P1** to **10xG-hSCC-P5**) are available at GEO: [GSE144240](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE144240).

### 10x scATAC-seq datasets

- Healthy and basal cell carcinoma human peripheral blood mononuclear cell scATAC-seq datasets (**10xC-hHPBMC** and **10xC-hPBMCBCC**) are available from [10x Genomics](https://www.10xgenomics.com/datasets).
- Mouse brain scATAC-seq datasets (**10xC-mBrain-0**, **10xC-mBrain-1**, and **10xC-mBrain-2**) are available from their original sources.

### Spatial transcriptomics datasets

- Slide-seqV2 mouse embryo dataset (**ssq-mEmb-33**) is available at GEO: [GSE197353](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE197353).