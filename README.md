# pytorch-apptainer

Template for publishing Apptainer Images

## Overview

This repository provides an automated build pipeline for creating and
publishing Apptainer/Singularity container images from PyTorch Docker images.

## Current Image

- **Source**: `docker.io/pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime`
- **Registry**: GitHub Container Registry (ghcr.io)
- **Tags**:
  - `ghcr.io/mul-cps/pytorch-apptainer/pytorch:2.4.0`
  - `ghcr.io/mul-cps/pytorch-apptainer/pytorch:latest`

## Usage

Pull and run the Apptainer image:

```bash
# Pull the image
apptainer pull oras://ghcr.io/mul-cps/pytorch-apptainer/pytorch:latest

# Run the container
apptainer run pytorch_latest.sif

# Execute Python with PyTorch
apptainer exec pytorch_latest.sif python -c "import torch; print(torch.__version__)"
```

## Build Pipeline

The GitHub Actions workflow automatically:
1. Builds the Apptainer image from the definition file
2. Pushes the image to GitHub Container Registry
3. Creates build artifacts for download

The workflow is triggered on:
- Push to `main` branch
- Pull requests
- Manual workflow dispatch
