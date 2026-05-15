# RADAR: Reference-guided Anomalous-cell Detection, Alignment, and Resolution

Detecting anomalous single cells from single-cell omics datasets is crucial for understanding disease-associated cellular heterogeneity and supporting precision medicine. Existing methods often address anomalous-cell detection, multi-sample alignment, and anomalous-cell characterization as separate tasks, making it difficult to systematically identify and compare anomalous cell populations across multiple samples and modalities.

We propose **RADAR**, a reference-guided generative framework for anomalous-cell detection, alignment, and resolution in multi-sample single-cell studies. RADAR integrates three key tasks into a unified workflow: detecting anomalous cells in target datasets, aligning multiple target samples into a common reference space, and resolving anomalous cells into biologically meaningful subtypes. Comprehensive evaluations on real-world single-cell datasets demonstrate RADAR's superior performance in anomalous-cell detection, multi-sample alignment, and fine-grained anomalous-cell resolution across multiple target datasets.

## Installation

RADAR is developed as a Python package. You will need to install Python, and the recommended version is Python 3.9.

You can download the package from GitHub and install it locally:

```commandline
git clone https://github.com/Catchxu/RADAR.git
cd RADAR/
pip install .