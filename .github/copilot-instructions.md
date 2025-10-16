# Copilot Instructions for pytorch-apptainer

## Repository Overview
This repository is a template for publishing Apptainer (formerly Singularity) container images optimized for PyTorch workloads.

## Key Technologies
- **Apptainer/Singularity**: Container platform for HPC and scientific computing
- **PyTorch**: Deep learning framework
- **Container Definition Files**: `.def` files that define Apptainer containers

## Development Guidelines

### Container Definition Files
- Apptainer definition files use `.def` extension
- Follow the Apptainer definition file format with sections: `Bootstrap`, `From`, `%post`, `%environment`, `%runscript`, etc.
- Optimize container builds for PyTorch dependencies and CUDA compatibility
- Include appropriate base images (e.g., `docker://pytorch/pytorch` or `docker://nvidia/cuda`)

### Best Practices
- **Build Optimization**: Combine related commands in `%post` section to reduce build time and complexity
- **Version Pinning**: Pin PyTorch and dependency versions for reproducibility
- **CUDA Compatibility**: Ensure CUDA versions match between base image and PyTorch
- **Documentation**: Document environment variables and expected usage in comments
- **Testing**: Include validation steps to verify PyTorch and GPU functionality

### Code Style
- Use clear, descriptive comments in definition files
- Organize installation commands logically (system packages, Python packages, configuration)
- Follow standard shell scripting best practices in `%post` sections

### Common Patterns
- Use `apt-get` with `-y` flag and `--no-install-recommends` to minimize image size
- Clean up package manager caches after installations
- Set `PYTHONUNBUFFERED=1` for better logging in containerized environments
- Include `%labels` section for metadata (version, maintainer, description)

## File Organization
- Definition files should be in the root directory or a `containers/` directory
- Name definition files descriptively (e.g., `pytorch-2.0-cuda11.8.def`)
- Keep build scripts and utilities in a `scripts/` directory if needed

## Building and Testing
- Build containers with: `apptainer build <image.sif> <definition.def>`
- Test PyTorch availability: `apptainer exec <image.sif> python -c "import torch; print(torch.__version__)"`
- Test GPU access: `apptainer exec --nv <image.sif> python -c "import torch; print(torch.cuda.is_available())"`

## Documentation
- Update README.md when adding new container definitions
- Include usage examples for each container image
- Document required hardware (GPU models, CUDA versions, etc.)
