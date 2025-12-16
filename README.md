# AutoR-Planner

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2506.24113-b31b1b.svg)](https://arxiv.org/abs/2506.24113)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**Autoregressive Diffusion World Model for Autonomous Driving**

Based on [Epona](https://kevin-thu.github.io/Epona/)

</div>

---

## Overview

AutoR-Planner is an autoregressive diffusion world model for autonomous driving, capable of generating high-quality driving videos and predicting future trajectories.

**Key Features:**
- 🎬 Long-horizon video generation (up to 120s / 600 frames)
- ⚡ KV-cache acceleration for STT encoding
- 🎨 Custom data testing support
- 📊 FVD/FID evaluation tools

---

## Performance

| Metric | Value |
|--------|-------|
| FID ↓ | 7.5 |
| FVD ↓ | 82.8 |
| Max Duration | 120s / 600 frames |

### Generation Speed (per frame)

| Module | Time |
|--------|------|
| STT (with KV-cache) | ~18ms |
| TrajDiT | ~380ms |
| VisDiT | ~1.8s |
| **Total** | **~2.2s/frame** |

---

## Installation

### Requirements
- Python 3.10+
- CUDA 12.1+
- GPU Memory ≥12GB

### Setup

```bash
# Create environment
conda create -n epona python=3.10 -y
conda activate epona

# Install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install opencv-python einops tqdm imageio imageio-ffmpeg
pip install safetensors timm accelerate deepspeed
pip install omegaconf addict mmengine matplotlib scipy scikit-learn
```

### Pre-trained Models

Download from [HuggingFace](https://huggingface.co/Kevin-thu/Epona) and place in `pretrained/`:

- `epona_nuplan.pkl` (4.9GB)
- `dcae_td_20000.pkl` (1.3GB)

---

## Quick Start

### Generate Custom Test Data

```bash
# Generate synthetic data
python generate_demo_data.py --num_videos 3 --num_frames 10

# Run inference
python scripts/test/test_demo.py \
    --exp_name "test" \
    --resume_path "pretrained/epona_nuplan.pkl" \
    --config "configs/dit_config_dcae_nuplan_cached.py"

# View results
ls test_videos/test/sliding_0/video.mp4
```

### Use NuPlan Dataset

```bash
python scripts/test/test_free_cached.py \
    --exp_name "nuplan_test" \
    --resume_path "pretrained/epona_nuplan.pkl" \
    --config "configs/dit_config_dcae_nuplan_cached.py" \
    --use_kv_cache
```

---

## Configuration

Key parameters in `configs/dit_config_dcae_nuplan_cached.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| test_video_frames | 50 | Generated frames (max 600) |
| downsample_fps | 5 | Video frame rate |
| condition_frames | 10 | Input condition frames |

---

## Evaluation

### FVD Metric

```bash
# Install dependencies
git clone https://github.com/piergiaj/pytorch-i3d.git
cd pytorch-i3d && pip install -e .

# Calculate FVD
python calculate_fvd.py \
    --real_videos /path/to/real \
    --fake_videos /path/to/generated \
    --from_images
```

---

## Project Structure

```
AutoR-Planner/
├── configs/              # Configuration files
├── models/               # Model implementations
├── dataset/              # Data loaders
├── scripts/              # Test scripts
├── pretrained/           # Pre-trained models
├── generate_demo_data.py # Data generation tool
└── calculate_fvd.py      # FVD evaluation tool
```

