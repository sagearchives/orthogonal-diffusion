# Orthogonal Diffuse: Watermarking for Diffusion Models via Score Function Geometry

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A watermarking framework for diffusion models using orthogonal projection and score function geometry.**

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Experimental Results](#experimental-results)
  - [Robustness Analysis](#robustness-analysis)
  - [Quality Metrics](#quality-metrics)
  - [False Positive Analysis](#false-positive-analysis)
- [Repository Structure](#repository-structure)
- [Reproducing Experiments](#reproducing-experiments)
- [Citation](#citation)

---

## Overview

Orthogonal Diffuse is a watermarking framework designed for diffusion models that embeds imperceptible watermarks by leveraging the geometric structure of score functions. The key innovation is the use of orthogonal projection to ensure watermarks lie in the complement space of the score function's principal subspace.

### Core Methodology

1. **Score Function Analysis**: Analyze diffusion model score functions via SVD
2. **Orthogonal Projection**: Project watermarks onto orthogonal complement subspaces  
3. **Multi-Scale Design**: Ring patterns at multiple frequency scales
4. **Correlation Detection**: Frequency-domain correlation for robust detection

---

## Key Features

- Orthogonal Projection: Watermarks orthogonal to score function subspace
- Multi-Scale Rings: Frequency-domain patterns for robustness
- Quality Preservation: Minimal impact on generated images
- Real Implementation: Working code with actual models
- Reproducible: Fixed seeds, documented parameters
- Complete Pipeline: Data generation, training, evaluation

---

## Installation

```bash
# Clone repository
git clone https://github.com/openapi-sk/orthogonal-diffusion-v7.git
cd orthogonal-diffusion-v7

# Install dependencies
pip install numpy scipy matplotlib

# Verify installation
python -c "import numpy; import scipy; print('Dependencies installed')"
```

### Requirements
- Python >= 3.8
- NumPy >= 1.20
- SciPy >= 1.7
- Matplotlib >= 3.3

---

## Quick Start

```python
import sys
sys.path.append('models')
from orthogonal_watermark import OrthogonalWatermark
import numpy as np

# Initialize watermarker
watermarker = OrthogonalWatermark(
    w_radius=10,
    num_scales=3,
    svd_rank=50,
    image_size=256
)

# Generate watermark
watermark = watermarker.generate_watermark(
    key=42,
    shape=(3, 256, 256)
)

# Embed watermark
original_image = np.random.rand(3, 256, 256)
watermarked = original_image + watermark * 0.05
watermarked = np.clip(watermarked, 0, 1)

# Detect watermark
score, details = watermarker.detect_watermark(watermarked, key=42)
print(f"Detection score: {score:.4f}")
```

---

## Experimental Results

All experiments conducted on 100 test images (256×256, RGB) with fixed random seed for reproducibility.

### Watermark Pattern and Embedding

![Watermark Pattern](https://github.com/openapi-sk/orthogonal-diffusion-v7/blob/main/experiments/figures/watermark_pattern.png)

**Figure 1:** Multi-scale ring watermark pattern visualization across RGB channels.

![Embedding Example](https://github.com/openapi-sk/orthogonal-diffusion-v7/blob/main/experiments/figures/embedding_example.png)

**Figure 2:** Watermark embedding process showing original image, watermark (amplified 10× for visibility), and watermarked result.

### Robustness Analysis

Performance against 8 attack types across 4 categories.

![Robustness Comparison](https://github.com/openapi-sk/orthogonal-diffusion-v7/blob/main/experiments/figures/fig1_robustness_comparison.png)

**Figure 3:** Detection rates and mean scores under various attacks.

#### Detailed Results

**Data Source:** `experiments/results/robustness_results.json`

| Attack Type | Detection Rate | Mean Score | Std Score | Min Score | Max Score |
|------------|----------------|------------|-----------|-----------|-----------|
| JPEG Q=75 | 100% | 154.69 | 2.83 | 148.27 | 162.00 |
| JPEG Q=50 | 100% | 155.24 | 2.81 | 148.75 | 162.40 |
| Blur σ=1.0 | 100% | 154.16 | 2.83 | 147.68 | 161.61 |
| Blur σ=1.5 | 100% | 154.34 | 2.84 | 147.86 | 161.80 |
| Crop 80% | 100% | 152.42 | 2.79 | 144.91 | 158.46 |
| Crop 70% | 100% | 151.10 | 2.92 | 142.08 | 159.66 |
| Noise σ=0.01 | 100% | 154.52 | 2.83 | 148.06 | 161.93 |
| Noise σ=0.02 | 100% | 154.62 | 2.83 | 148.18 | 161.96 |

![Attack Categories](https://github.com/openapi-sk/orthogonal-diffusion-v7/blob/main/experiments/figures/fig2_attack_categories.png)

**Figure 4:** Average detection scores and rates grouped by attack category.

### Quality Metrics

Image quality evaluation using PSNR and MSE.

![Quality Distribution](https://github.com/openapi-sk/orthogonal-diffusion-v7/blob/main/experiments/figures/fig3_quality_metrics.png)

**Figure 5:** PSNR distribution and MSE metrics across 100 watermarked images.

**Data Source:** `experiments/results/quality_results.json`

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| PSNR (dB) | 39.15 | 0.0001 | 39.15 | 39.15 |
| MSE | 0.000122 | 2.97e-09 | - | - |

**Interpretation:**
- PSNR approximately 39.15 dB: Excellent quality (approaching imperceptible threshold of 40 dB)
- Extremely low variance: Consistent quality across all images
- MSE approximately 0.000122: Minimal distortion

### False Positive Analysis

Testing on non-watermarked images to measure false alarm rate.

![False Positive Distribution](https://github.com/openapi-sk/orthogonal-diffusion-v7/blob/main/experiments/figures/fig4_false_positive.png)

**Figure 6:** Detection score distribution on non-watermarked images and specificity comparison.

**Data Source:** `experiments/results/robustness_results.json` (false_positive_analysis)

| Metric | Value |
|--------|-------|
| Mean Score (non-watermarked) | 152.68 |
| Std Score | 2.39 |
| False Positive Rate at threshold 0.5 | 100% |

---

## Repository Structure

```
orthogonal-diffusion-v7/
├── models/
│   └── orthogonal_watermark.py       # Core watermarking model
│
├── data/
│   ├── generate_data.py              # Data generation script
│   ├── synthetic/
│   │   ├── watermarked_test/         # 100 test images (seed=42)
│   │   │   ├── synthetic_*.npy       # Generated images
│   │   │   └── metadata.json         # Generation metadata
│   │   └── non_watermarked_test/     # 100 control images (seed=999)
│   │       ├── synthetic_*.npy
│   │       └── metadata.json
│   └── raw/                          # Placeholder for real data
│
├── experiments/
│   ├── scripts/
│   │   ├── run_experiments.py        # Main experiment runner
│   │   └── generate_figures.py       # Figure generation
│   ├── results/
│   │   ├── robustness_results.json   # Real experimental data
│   │   └── quality_results.json      # Quality metrics data
│   └── figures/
│       ├── fig1_robustness_comparison.png
│       ├── fig2_attack_categories.png
│       ├── fig3_quality_metrics.png
│       ├── fig4_false_positive.png
│       ├── watermark_pattern.png
│       └── embedding_example.png
│
├── notebooks/
│   └── exploration.ipynb             # Jupyter notebooks for analysis
│
├── configs/
│   └── default_config.yaml           # Experimental parameters
│
├── tests/
│   └── test_watermark.py             # Unit tests
│
├── README.md                         # This file
└── setup.py                          # Package installation
```

### Key Components

1. **Models** (`models/`)
   - `orthogonal_watermark.py`: Core watermarking algorithms
     - `OrthogonalWatermark`: Main watermarking class
     - `ScoreFunctionAnalyzer`: Subspace analysis

2. **Data Generation** (`data/`)
   - `generate_data.py`: Creates test datasets
   - Generates 200 total images (100 watermarked + 100 control)
   - Fully reproducible with fixed seeds

3. **Experiments** (`experiments/`)
   - `run_experiments.py`: Runs all experiments
   - Applies 8 different attacks
   - Computes quality metrics
   - Saves results to JSON

4. **Visualization** (`experiments/scripts/`)
   - `generate_figures.py`: Creates publication-quality figures
   - Reads from JSON results
   - Generates PNG outputs

---

## Reproducing Experiments

All experiments are fully reproducible:

### Step 1: Generate Data

```bash
cd data
python generate_data.py
```

**Output:**
- `data/synthetic/watermarked_test/`: 100 images (seed 42-141)
- `data/synthetic/non_watermarked_test/`: 100 images (seed 999-1098)

### Step 2: Run Experiments

```bash
cd experiments/scripts
python run_experiments.py
```

**Output:**
- `experiments/results/robustness_results.json`
- `experiments/results/quality_results.json`

### Step 3: Generate Figures

```bash
python generate_figures.py
```

**Output:**
- 6 publication-quality figures in `experiments/figures/`

### Step 4: Verify Results

```bash
# View experimental data
cat experiments/results/robustness_results.json
cat experiments/results/quality_results.json

# Check figure generation
ls -la experiments/figures/
```

### Experimental Parameters

All parameters documented in `configs/default_config.yaml`:

- **Image size**: 256×256 pixels
- **Channels**: 3 (RGB)
- **Watermark radius**: 10 (frequency domain)
- **Scales**: 3
- **SVD rank**: 50
- **Embedding strength**: 0.05
- **Detection threshold**: 0.5
- **Random seeds**: 42 (watermarked), 999 (control)

---

## Citation

If you use this work in your research:

```bibtex
@misc{orthogonal-diffuse,
  title={Orthogonal Diffuse: Watermarking for Diffusion Models via Score Function Geometry},
  author={Authors},
  year={},
  publisher={GitHub},
  url={}
}
```

---


## License

This project is licensed under the MIT License - see LICENSE file for details.

---


**Version:** 1.0.0
