# XLSR - Extremely Lightweight Quantization Robust Real-Time Single-Image Super Resolution for Mobile Devices

Welcome to the Pytorch-Lightning implementation of the research paper **"Extremely Lightweight Quantization Robust Real-Time Single-Image Super Resolution for Mobile Devices (XLSR)"**. XLSR is a cutting-edge solution engineered to deliver exceptional image super-resolution while remaining lightweight, quantization-robust, and optimized for real-time execution on mobile devices.

---

## Features
- **Lightweight Model**: Designed for minimal memory footprint and computational efficiency.
- **Quantization Robustness**: Ensures high performance even after quantization, enabling seamless deployment on mobile hardware.
- **Real-Time Performance**: Provides low-latency performance, ideal for mobile and embedded systems.

---

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Model Architecture](#model-architecture)
4. [Dataset](#dataset)
5. [Training](#training)
6. [Results](#results)
7. [References](#references)

---

## Installation
### Requirements
To get started, ensure the following dependencies are installed:

- Python 3.10 or later
- PyTorch 2.3 or later
- PyTorch Lightning (latest version)
- Pillow
- NumPy

---

## Usage

### Quick Start Example:
```bash
see test.ipynb
```

---

## Model Architecture
XLSR leverages an efficient design composed of:
- **Efficient Residual Blocks**: Enhances feature extraction with minimal overhead.
- **Pixel Shuffle Layers**: Upscales images efficiently with reduced computational complexity.

---

## Dataset
XLSR supports the following datasets:
- **DIV2K** - A high-quality image super-resolution dataset.

---

## Training
To train the XLSR model, execute:
```bash
python main.py
```

---

## Results
### Benchmark Results
| Model         | Parameters (K) | PSNR (dB) |
|---------------|----------------|-----------|
| XLSR          | 20             | 29.58     |

---

## References
- Original Paper: [arXiv:2105.10288 ](https://arxiv.org/abs/2105.10288v1)
- Journal reference: IEEE Computer Vision Pattern Recognition Workshops (Mobile AI 2021 Workshop)

