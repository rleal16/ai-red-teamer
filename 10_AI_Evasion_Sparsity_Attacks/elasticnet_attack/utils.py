#!/usr/bin/env python3
from __future__ import annotations
import argparse, base64, io, json, time
from dataclasses import dataclass
import numpy as np
import requests
from PIL import Image
import torch
import torch.nn as nn
import os


def set_seed(seed: int = 1337) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class SimpleClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def mnist_normalize(x01: torch.Tensor) -> torch.Tensor:
    return (x01 - MNIST_MEAN) / MNIST_STD

def x01_from_b64_png(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("L")
    x = np.asarray(img, dtype=np.float32) / 255.0
    return np.clip(x, 0.0, 1.0)


def b64_png_from_x01(x2d: np.ndarray) -> str:
    x255 = np.clip((x2d * 255.0).round(), 0, 255).astype(np.uint8)
    img = Image.fromarray(x255, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")

@dataclass
class Challenge:
    label: int
    beta: float
    elastic_max: float
    l2_max: float
    l1_max: float
    sample_index: int
    x01: np.ndarray


def fetch_challenge(host: str) -> Challenge:
    
    r = requests.get(f"{host}/challenge", timeout=10)
    r.raise_for_status()
    p = r.json()
    x2d = x01_from_b64_png(p["image_b64"])
    x4d = x2d[None, None, ...].astype(np.float32)
    return Challenge(
        label=int(p["label"]),
        beta=float(p["beta"]),
        elastic_max=float(p["elastic_max"]),
        l2_max=float(p["l2_max"]),
        l1_max=float(p["l1_max"]),
        sample_index=int(p["sample_index"]),
        x01=x4d,
    )


def load_model(weights: str, device: torch.device) -> nn.Module:
    m = SimpleClassifier().to(device).eval()
    m.load_state_dict(torch.load(weights, map_location=device))
    return m


def _post_submit(host: str, b64: str) -> requests.Response:
    """Submit base64 PNG and return the raw response."""
    return requests.post(f"{host}/submit", json={"image_b64": b64}, timeout=30)


def _post_submit_retry(
    host: str, b64: str, retries: int = 3, delay: float = 0.5
) -> requests.Response:
    """Submit with basic retry to tolerate transient connection resets."""
    last = None
    for i in range(max(1, retries)):
        try:
            resp = _post_submit(host, b64)
            return resp
        except requests.RequestException as e:
            last = e
            time.sleep(delay)
    if last is not None:
        raise last