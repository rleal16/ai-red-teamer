#!/usr/bin/env python3
from __future__ import annotations
import argparse
import base64
import io
import json
from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import requests
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# MNIST normalization constants used by the pretrained model
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

class MNISTClassifier(nn.Module):
    """LeNet-5 style network mirroring the FastAPI server model."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return log-softmax scores for the supplied MNIST batch."""
        x = self.act(self.conv1(x))  # (B,6,24,24)
        x = self.pool(x)  # (B,6,12,12)
        x = self.act(self.conv2(x))  # (B,16,8,8)
        x = self.pool(x)  # (B,16,4,4)
        x = torch.flatten(x, 1)  # (B,256)
        x = self.act(self.fc1(x))  # (B,120)
        x = self.act(self.fc2(x))  # (B,84)
        x = self.fc3(x)  # (B,10)
        return F.log_softmax(x, dim=1)

def mnist_normalize(x01: torch.Tensor) -> torch.Tensor:
    """Normalize a [0,1] tensor to MNIST stats for the classifier."""
    return (x01 - MNIST_MEAN) / MNIST_STD

def png_from_x01(x01: np.ndarray) -> str:
    """Encode a [0,1] grayscale image (28,28) to base64 PNG string."""
    x255 = np.clip((x01 * 255.0).round(), 0, 255).astype(np.uint8)
    img = Image.fromarray(x255, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def x01_from_b64_png(b64: str) -> np.ndarray:
    """Decode base64 PNG to [0,1] numpy array of shape (28,28)."""
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("L")
    if img.size != (28, 28):
        raise ValueError("Expected 28x28 PNG")
    x = np.asarray(img, dtype=np.float32) / 255.0
    return np.clip(x, 0.0, 1.0)

@dataclass
class Challenge:
    target_class: int
    l0_budget: int
    original_label: int
    sample_index: int
    x01: np.ndarray  # (1,1,28,28)

def fetch_challenge(host: str, retries: int = 30, delay: float = 1.0) -> Challenge:
    """Fetch challenge with simple retry/backoff to tolerate startup races."""
    print("Fetching challenge")
    last_err = None
    for _ in range(max(1, retries)):
        try:
            r = requests.get(f"{host}/challenge", timeout=5)
            r.raise_for_status()
            payload = r.json()
            break
        except Exception as e:
            last_err = e
            time.sleep(delay)
    else:
        raise RuntimeError(
            f"Failed to connect to {host}/challenge: {last_err}"
        ) from last_err
    x2d = x01_from_b64_png(payload["image_b64"])  # (28,28)
    x4d = x2d[None, None, ...]
    print("Challenge Fetched")
    return Challenge(
        target_class=int(payload["target_class"]),
        l0_budget=int(payload["l0_budget"]),
        original_label=int(payload["original_label"]),
        sample_index=int(payload["sample_index"]),
        x01=x4d.astype(np.float32),
    )

def load_model(host: str, weights_path: Optional[str] = None) -> MNISTClassifier:
    """Load the JSMA classifier using either a local file or the lab endpoint."""
    print("Loading Model")
    model = MNISTClassifier()
    if weights_path:
        state = torch.load(weights_path, map_location=torch.device("cpu"))
    else:
        resp = requests.get(f"{host}/weights", timeout=30)
        try:
            resp.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download weights from {host}/weights"
            ) from exc
        buffer = io.BytesIO(resp.content)
        state = torch.load(buffer, map_location=torch.device("cpu"))
    model.load_state_dict(state)
    model.eval()
    print("Model Loaded")
    return model