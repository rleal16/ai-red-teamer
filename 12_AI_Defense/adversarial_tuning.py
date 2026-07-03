"""Ensure required packages are installed; install via pip if missing.
   Check CUDA availability and display GPU information."""

import os
import subprocess
import sys
from pathlib import Path

# Enable line-buffered output for real-time progress updates
sys.stdout.reconfigure(line_buffering=True)

# set huggingface cache to local directory for portability
LAB_DIR = Path(".")
os.environ["HF_HOME"] = str(LAB_DIR / "hf_cache")

REQUIRED_PACKAGES = [
    "unsloth",  # efficient LoRA tuning
    "transformers",  # for the base model architecture
    "trl",  # for the supervised fine-tuning trainer
    "datasets",
    "accelerate",
    "bitsandbytes",
]


def ensure_packages(packages=None):
    """Check if packages are importable; pip install any that are missing."""
    packages = packages or REQUIRED_PACKAGES
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )


def check_cuda_and_gpu():
    """Confirm CUDA availability and display GPU information.
    If CUDA is not available, print guidance to verify drivers and CUDA toolkit."""
    import torch

    print("=" * 60)
    print("CUDA / GPU check")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    if torch.version.cuda is not None:
        print(f"PyTorch CUDA version: {torch.version.cuda}")
    else:
        print("PyTorch CUDA version: N/A (CPU or non-CUDA build)")
    print()

    if torch.cuda.is_available():
        print("CUDA is available.")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            total_mem = props.total_memory / (1024**3)
            print(f"  GPU {i}: {name}")
            print(f"    Compute capability: {props.major}.{props.minor}")
            print(f"    Total memory: {total_mem:.2f} GB")
        print("=" * 60)
        return True

    print("CUDA is NOT available.")
    print()
    print("To use GPU acceleration, verify:")
    print("  1. NVIDIA GPU drivers are in x§stalled and up to date.")
    print("  2. CUDA toolkit is installed and matches your PyTorch build.")
    print("  3. PyTorch was installed with CUDA support for your CUDA version, e.g.:")
    print("       pip install torch --index-url https://download.pytorch.org/whl/cu121")
    print()
    print("Check compatibility: https://pytorch.org/get-started/locally/")
    print("=" * 60)
    return False


if __name__ == "__main__":
    ensure_packages()
    check_cuda_and_gpu()
