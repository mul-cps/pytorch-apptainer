# 🧩 pytorch-apptainer

[![Build](https://github.com/mul-cps/pytorch-apptainer/actions/workflows/build.yml/badge.svg)](https://github.com/mul-cps/pytorch-apptainer/actions/workflows/build.yml)
[![Container Registry](https://img.shields.io/badge/registry-ghcr.io-blue?logo=github)](https://ghcr.io/mul-cps/pytorch-apptainer)
[![Apptainer](https://img.shields.io/badge/Apptainer-%F0%9F%94%97-green)](https://apptainer.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

> 🧠 Template for publishing and using **Apptainer/Singularity** images derived from official **PyTorch** Docker releases.

---

## 🧭 Overview

This repository provides an automated build pipeline that creates and publishes Apptainer (`.sif`) images based on official **PyTorch CUDA** runtime containers.
The images are hosted on **GitHub Container Registry (GHCR)** and can be used directly on HPC clusters, where Docker access is typically restricted.

---

## 🧱 Current Image

| Component        | Value                                                                          |
| ---------------- | ------------------------------------------------------------------------------ |
| **Base image**   | `docker.io/pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime`                      |
| **Registry**     | [ghcr.io/mul-cps/pytorch-apptainer](https://ghcr.io/mul-cps/pytorch-apptainer) |
| **Tags**         | `2.4.0`, `latest`                                                              |
| **Architecture** | `linux/amd64`                                                                  |
| **CUDA**         | 12.1                                                                           |
| **cuDNN**        | 9                                                                              |

---

## ⚙️ Recommended Usage Workflow

Directly pulling from the registry on the cluster can **hang or fail** due to firewall or proxy limits.
Instead, build or download locally, then transfer the image to the cluster.

### 🔧 Option A — Build locally

```bash
apptainer build pytorch_24.06.sif apptainer.def
scp pytorch_24.06.sif user@hpc-login1:/home/user/
```

### ⬇️ Option B — Download from GHCR via ORAS

```bash
apptainer pull oras://ghcr.io/mul-cps/pytorch-apptainer/pytorch:latest
scp pytorch_24.06.sif user@hpc-login1:/home/user/
```

### ▶️ Run on the cluster

```bash
apptainer exec --nv ~/pytorch_24.06.sif python -c "import torch; print(torch.__version__)"
```

> `--nv` passes through host GPU devices and drivers into the container runtime.

---

# 🖥️ Slurm 101 – Cluster Basics

**Slurm** is the workload manager used on most HPC clusters.

Typical workflow:

1. **Connect** via SSH to a login node
2. **Prepare your files** (scripts, data, `.sif` image)
3. **Submit jobs** with `sbatch job.sbatch`
4. **Monitor** jobs with `squeue -u $USER`
5. **Inspect** outputs via `.out` / `.err` files

### GPU Scheduling

* **Partitions** (e.g. `p2gpu`) define available hardware.
* **QoS** refines access/priority.
* GPUs must be requested explicitly (`--gpus-per-node` or `--gres=gpu:X`).
* For PyTorch DDP, use **one task per GPU** for best performance.

---

# 🔑 Connecting & Moving the Image

```bash
# 1) Connect to the login node
ssh user@hpc-login1

# 2) Copy your local image to the cluster
scp ./pytorch_24.06.sif user@hpc-login1:/home/user/
```

---

# 🧰 Load Apptainer Module

```bash
module purge
module load apptainer/1.1.7-gcc-13.2.0-36exqw4
apptainer --version
```

> `module purge` clears conflicting environments.
> Always load the specific module version used in your workflow for reproducibility.

---

# ✅ Quick Sanity Check

Verify GPU visibility and multi-node allocation:

```bash
srun -N2 -p p2gpu --qos=p2gpu --wait=0 nvidia-smi
```

This checks that

* Two nodes were allocated
* GPUs are visible on each node
* The scheduler correctly launches parallel commands

---

# 🧪 Multi-Node PyTorch DDP Benchmark

Save as `torch_ddp_bench.sbatch`:

```bash
#!/bin/bash
#SBATCH -J torch-ddp-bench
#SBATCH -p p2gpu
#SBATCH --qos=p2gpu
#SBATCH -N 2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH -o ddp_bench_%j.out
#SBATCH -e ddp_bench_%j.err
#SBATCH --exclusive

echo "Nodes: $SLURM_NODELIST"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "GPUs per node (requested): $SLURM_JOB_GPUS_PER_NODE"

# --- Environment setup ---
module purge
module load apptainer/1.1.7-gcc-13.2.0-36exqw4
export OMP_NUM_THREADS=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=0

# --- Rendezvous for PyTorch DDP ---
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29500
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

# --- Multi-node DDP benchmark ---
srun --mpi=pmix \
  apptainer exec --nv ~/pytorch_24.06.sif \
  python - <<'PY'
import os, time, torch, torch.distributed as dist

rank       = int(os.environ["SLURM_PROCID"])
world_size = int(os.environ["SLURM_NTASKS"])
local_rank = int(os.environ.get("SLURM_LOCALID", 0))
torch.cuda.set_device(local_rank)

dist.init_process_group(
    backend="nccl",
    init_method="env://",
    rank=rank,
    world_size=world_size
)

if rank == 0:
    print(f"Torch: {torch.__version__}")
    print(f"NCCL backend initialized with world_size={world_size}")

n = 4096
iters = 50
a = torch.randn((n,n), device="cuda", dtype=torch.float32)
b = torch.randn((n,n), device="cuda", dtype=torch.float32)

torch.cuda.synchronize(); t0 = time.time()
for _ in range(iters):
    c = a @ b
torch.cuda.synchronize()
dt = time.time() - t0

flops_rank = iters * 2 * (n**3) / dt
t = torch.tensor([flops_rank], device="cuda", dtype=torch.float64)
dist.all_reduce(t, op=dist.ReduceOp.SUM)
flops_global = t.item()

dt_t = torch.tensor([dt], device="cuda", dtype=torch.float64)
dist.all_reduce(dt_t, op=dist.ReduceOp.MAX)
dt_global = dt_t.item()

if rank == 0:
    print(f"{iters}x {n}x{n} GEMM:")
    print(f"  Max loop time across ranks: {dt_global:.3f}s")
    print(f"  Per-rank: ~{flops_rank/1e12:.2f} TFLOP/s")
    print(f"  Aggregated: ~{flops_global/1e12:.2f} TFLOP/s")

dist.destroy_process_group()
PY
```

---

## 🧩 Explanation of Key Steps

| Step                                    | Purpose                                                 |
| --------------------------------------- | ------------------------------------------------------- |
| `--gpus-per-node` + `--ntasks-per-node` | One rank per GPU → simpler device mapping               |
| `MASTER_ADDR/PORT`                      | Rendezvous endpoint for all distributed ranks           |
| `--mpi=pmix`                            | Exports rank/env info (`SLURM_PROCID`, `SLURM_LOCALID`) |
| `apptainer exec --nv`                   | Mounts host GPU devices & drivers inside the container  |
| `OMP_NUM_THREADS=1`                     | Prevents CPU oversubscription                           |
| `NCCL_*` envs                           | Control verbosity, async errors, and comms stability    |

---

## ▶️ How to Run the Job

```bash
sbatch torch_ddp_bench.sbatch          # Submit job
squeue -u $USER                        # Monitor queue
tail -f ddp_bench_<JOBID>.out          # View output live
```

If you see NCCL timeouts or hangs:

* Test first with `-N 1` (single node)
* Ensure `--gpus-per-node` = `--ntasks-per-node`
* Set `NCCL_DEBUG=INFO` for diagnostics

---

## 🧩 Interactive Debug Mode

```bash
salloc -N1 -p p2gpu --qos=p2gpu --gpus-per-node=1 --time=00:30:00
srun --ntasks=1 --mpi=pmix \
  apptainer exec --nv ~/pytorch_24.06.sif \
  python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0))"
```

---

## 🧰 Repository Structure

```
├── apptainer.def        # Definition file for container build
├── .github/workflows/   # CI build & publish pipeline
├── README.md            # This document
└── LICENSE              # MIT license
```

---

## 🏗️ Build Pipeline (Continuous Integration)

The GitHub Actions workflow automatically:

1. Builds the Apptainer image from `apptainer.def`
2. Pushes the `.sif` image to GHCR via ORAS
3. Publishes `.sif` build artifacts for download

Triggered on:

* Push to `main`
* Pull requests
* Manual workflow dispatch

---

## 📚 References

* [Apptainer Documentation](https://apptainer.org/docs/)
* [PyTorch Docker Images](https://hub.docker.com/r/pytorch/pytorch)
* [ORAS Project](https://oras.land/)
* [GitHub Container Registry Docs](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
