#!/usr/bin/env python3

import os
from unittest import loader

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from safetensors.torch import save_file
from tqdm import tqdm
from opacus import PrivacyEngine, privacy_engine
from opacus.validators import ModuleValidator


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_SEED = 1337
BATCH_SIZE = 256
DP_EPOCHS = 20
DP_LR = 0.1
MAX_GRAD_NORM = 1.0
DELTA = 1e-5

# The key parameter: use ε=6 to balance accuracy and privacy
TARGET_EPSILON = 6.0

# Set reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output directories
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# SVHN normalization values
SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD = (0.1980, 0.2010, 0.1970)


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================


class SVHNCNN(nn.Module):
    """CNN for SVHN classification (compatible with Opacus)."""

    def __init__(self):
        super(SVHNCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# =============================================================================
# DATA LOADING
# =============================================================================


def get_svhn_loaders(batch_size=256, download=True):
    """Load SVHN dataset."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(SVHN_MEAN, SVHN_STD)]
    )

    train_dataset = datasets.SVHN(
        "data", split="train", download=download, transform=transform
    )
    test_dataset = datasets.SVHN(
        "data", split="test", download=download, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_dataset, test_dataset, train_loader, test_loader


def evaluate_accuracy(model, data_loader, device):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100.0 * correct / total


def compute_mia_advantage(model, train_loader, test_loader, device="cpu", num_samples=2000):
    """Return (mia_accuracy, mia_advantage) for a confidence-threshold MIA baseline."""
    def get_confidences(loader, max_samples):
        confs = []
        model.eval()
        processed = 0
        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                max_prob = F.softmax(logits, dim=1).max(dim=1).values
                confs.append(max_prob.cpu().numpy())
                
                processed += imgs.size(0) 
                if processed >= max_samples:
                    break
        
        return np.concatenate(confs)[:max_samples]

    best_acc = 0.5
    members_conf = get_confidences(train_loader, num_samples)
    nonmembers_conf = get_confidences(test_loader, num_samples)

    n = min(len(members_conf), len(nonmembers_conf))
    all_confs = np.hstack((members_conf[:n], nonmembers_conf[:n]))
    all_labels = np.concatenate([np.ones(n), np.zeros(n)])
    thresholds = np.percentile(all_confs, np.linspace(0, 100, 500))

    for t in thresholds:
        
        best_acc = max(
            max(
                np.mean((all_confs > t) == all_labels),
                np.mean((all_confs < t) == all_labels)
            ),
            best_acc
        )
        
    
    return best_acc, best_acc - 0.5
                


        

# =============================================================================
# TRAINING (learner implements)
# =============================================================================


def train_dp_sgd(device, train_loader, test_loader):
    model = SVHNCNN().to(device)
    model = ModuleValidator.fix(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=DP_LR, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    privacy_engine = PrivacyEngine(accountant='rdp')
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_epsilon=TARGET_EPSILON,
        target_delta=DELTA,
        epochs=DP_EPOCHS,
        max_grad_norm=MAX_GRAD_NORM
    )

    # training loop scaffold (learner implements optimization steps)
    model.train()
    for epoch in range(DP_EPOCHS):
        for img, lbls in train_loader:
            optimizer.zero_grad(set_to_none=True)
            img = img.to(device)
            lbls = lbls.to(device)
            logits = model(img)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
        eps = privacy_engine.get_epsilon(DELTA)
        print(f"Epoch {epoch+1}: eps={eps:.3f}")
            

    #mia_acc, mia_adv = compute_mia_advantage(model, train_loader, test_loader, device)
    #_ = mia_acc, mia_adv

    return model, privacy_engine


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 80)
    print("  DP-SGD PRIVACY (SVHN)")
    print("=" * 80)
    print(f"\nDevice: {device}")
    print(f"Target epsilon: {TARGET_EPSILON}")
    print(f"Required: accuracy >= 55%, MIA advantage <= 5%")

    print("\nLoading SVHN dataset...")
    train_dataset, test_dataset, train_loader, test_loader = get_svhn_loaders(
        batch_size=BATCH_SIZE, download=True
    )

    print(f"Training samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")

    print("\n" + "=" * 80)
    print(f"  TRAINING: DP-SGD MODEL (Target ε={TARGET_EPSILON})")
    print("=" * 80)

    # Fresh data loaders for DP training
    _, _, train_loader_dp, test_loader_dp = get_svhn_loaders(
        batch_size=BATCH_SIZE, download=False
    )

    dp_model, privacy_engine = train_dp_sgd(device, train_loader_dp, test_loader_dp)

    final_epsilon = privacy_engine.get_epsilon(DELTA)
    print(f"\nFinal privacy guarantee: (ε={final_epsilon:.2f}, δ={DELTA})")

    # Evaluate
    train_acc = evaluate_accuracy(dp_model, train_loader_dp, device)
    test_acc = evaluate_accuracy(dp_model, test_loader_dp, device)

    print(f"\nDP Model Performance:")
    print(f"  Training accuracy: {train_acc:.2f}%")
    print(f"  Test accuracy: {test_acc:.2f}%")
    print(f"  Overfitting gap: {train_acc - test_acc:.2f}%")

    # MIA evaluation
    print("\n" + "=" * 80)
    print("  MEMBERSHIP INFERENCE EVALUATION")
    print("=" * 80)

    mia_acc, mia_adv = compute_mia_advantage(
        dp_model, train_loader_dp, test_loader_dp, device
    )

    print(f"\nMIA Results:")
    print(f"  Attack accuracy: {mia_acc:.4f}")
    print(f"  Attack advantage: {mia_adv:.4f} ({mia_adv*100:.2f}%)")

    # Save model
    model_path = os.path.join(MODELS_DIR, "dp_model.safetensors")
    save_file(dp_model._module.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    # =============================================================================
    # VALIDATION CHECK
    # =============================================================================

    print("\n" + "=" * 80)
    print("  VALIDATION CHECK")
    print("=" * 80)

    accuracy_pass = test_acc >= 55.0
    mia_pass = mia_adv <= 0.05

    print(f"\nChallenge Requirements:")
    print(
        f"  Accuracy >= 55%:       {test_acc:.2f}% {'✓ PASS' if accuracy_pass else '✗ FAIL'}"
    )
    print(
        f"  MIA advantage <= 5%: {mia_adv*100:.2f}% {'✓ PASS' if mia_pass else '✗ FAIL'}"
    )

    if accuracy_pass and mia_pass:
        print("\n" + "=" * 80)
        print("  SOLUTION READY FOR SUBMISSION")
        print("=" * 80)
        print(f"\nSubmit {model_path} to the validator to get the flag!")
    else:
        print("\n" + "=" * 80)
        print("  SOLUTION NOT YET PASSING")
        print("=" * 80)
        print("\nConsider adjusting epsilon or other hyperparameters.")

    base_url = os.environ.get("BASE_URL")
    if base_url:
        import requests

        validate_url = base_url.rstrip("/") + "/validate"
        with open(model_path, "rb") as f:
            response = requests.post(
                validate_url,
                files={
                    "model": (
                        "dp_model.safetensors",
                        f,
                        "application/octet-stream",
                    )
                },
            )
        print(response.text)


if __name__ == "__main__":
    main()
