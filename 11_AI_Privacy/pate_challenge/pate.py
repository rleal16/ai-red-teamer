#!/usr/bin/env python3

import os
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from safetensors.torch import save_file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from tqdm import tqdm

# =============================================================================
# CONFIGURATION - OPTIMIZED FOR EMNIST LETTERS (26 CLASSES)
# =============================================================================

RANDOM_SEED = 1337
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

DATASET_CONFIG = {"name": "emnist_letters", "num_classes": 26, "num_features": 784}

TEACHER_CONFIG = {
    "num_teachers": 25,
    "hidden_layers": [256, 128],
    "dropout": 0.2,
    "epochs": 25,
    "batch_size": 64,
    "learning_rate": 0.001,
}

AGGREGATION_CONFIG = {
    "noise_scale": 1.0,
    "num_student_queries": 20000,
    "confident_threshold": 15,
}

STUDENT_CONFIG = {
    "hidden_layers": [256, 128],
    "dropout": 0.1,
    "epochs": 120,
    "batch_size": 64,
    "learning_rate": 0.001,
}

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class MLP(nn.Module):
    """Multi-Layer Perceptron for EMNIST Letters classification."""

    def __init__(self, input_size=784, hidden_layers=None, num_classes=26, dropout=0.2):
        super(MLP, self).__init__()
        if hidden_layers is None:
            hidden_layers = [256, 128]

        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        prev_size = input_size
        for hidden_size in hidden_layers:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.dropouts.append(nn.Dropout(dropout))
            prev_size = hidden_size

        self.output = nn.Linear(prev_size, num_classes)

    def forward(self, x):
        for layer, dropout in zip(self.layers, self.dropouts):
            x = F.relu(layer(x))
            x = dropout(x)
        return self.output(x)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_dataloader(X, y, batch_size, shuffle=True):
    """Create a DataLoader from numpy arrays."""
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(model, train_loader, val_loader, device, epochs, learning_rate, verbose=False):
    """Train a model."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    return model


def evaluate_accuracy(model, data_loader, device):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()
    return 100.0 * correct / total


def get_model_predictions(model, X, device):
    """Get model predictions."""
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = F.softmax(outputs, dim=1)
    return probs.cpu().numpy()


def get_teacher_votes(teachers, X, device):
    """Get vote counts from all teachers."""
    num_samples = X.shape[0]
    num_classes = DATASET_CONFIG["num_classes"]
    votes = np.zeros((num_samples, num_classes), dtype=np.int32)

    for teacher in teachers:
        preds = get_model_predictions(teacher, X, device)
        predictions = np.argmax(preds, axis=1)
        for i, pred in enumerate(predictions):
            votes[i, pred] += 1

    return votes


def pate_core_aggregation(votes: np.ndarray, noise_scale: float, threshold: float):
    num_samples, _ = votes.shape
    confident_mask = votes.max(axis=1) >= threshold
    labels = np.full((num_samples,), -1, dtype=np.int64)
    votes_cpy = votes.astype(np.float64).copy()

    votes_cpy[confident_mask] += np.random.laplace(0.0, noise_scale, size=votes_cpy[confident_mask].shape)
    labels[confident_mask] = np.argmax(votes_cpy[confident_mask], axis=1)

    return labels, confident_mask



def compute_mia_advantage_quick(model, X_members, X_nonmembers, device):
    """Compute MIA advantage."""
    model.eval()

    member_probs = get_model_predictions(model, X_members, device)
    member_conf = np.max(member_probs, axis=1)

    nonmember_probs = get_model_predictions(model, X_nonmembers, device)
    nonmember_conf = np.max(nonmember_probs, axis=1)

    n = min(len(member_conf), len(nonmember_conf), 2000)
    all_conf = np.concatenate([member_conf[:n], nonmember_conf[:n]])
    all_labels = np.concatenate([np.ones(n), np.zeros(n)])

    thresholds = np.percentile(all_conf, np.linspace(0, 100, 500))
    best_acc = 0.5

    for t in thresholds:
        acc = max(
            np.mean((all_conf >= t) == all_labels),
            np.mean((all_conf < t) == all_labels),
        )
        best_acc = max(best_acc, acc)

    return best_acc - 0.5


# =============================================================================
# MAIN PIPELINE
# =============================================================================

print("=" * 60)
print("PATE PRIVACY (EMNIST Letters)")
print("Scaffold: implement core PATE aggregation logic")
print("=" * 60)
print(f"\nDevice: {DEVICE}")
print("Required: accuracy >= 80%, MIA advantage <= 3%")

print("\n" + "=" * 60)
print("PHASE 1: Loading EMNIST Letters Dataset")
print("=" * 60)

train_dataset = datasets.EMNIST("data", split="letters", train=True, download=False)
test_dataset = datasets.EMNIST("data", split="letters", train=False, download=False)

X_train = train_dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
y_train = train_dataset.targets.numpy() - 1
X_test = test_dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
y_test = test_dataset.targets.numpy() - 1

X_private, X_public, y_private, y_public = train_test_split(
    X_train, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train
)
X_holdout, y_holdout = X_test, y_test

print(f"Private (teacher training): {len(X_private)} samples")
print(f"Public (student queries):   {len(X_public)} samples")
print(f"Holdout (evaluation):       {len(X_holdout)} samples")
print(f"Number of classes:          {DATASET_CONFIG['num_classes']}")

scaler = StandardScaler()
X_private_norm = scaler.fit_transform(X_private)
X_public_norm = scaler.transform(X_public)
X_holdout_norm = scaler.transform(X_holdout)

print("\n" + "=" * 60)
print("PHASE 2: Training Teacher Ensemble")
print("=" * 60)

num_teachers = TEACHER_CONFIG["num_teachers"]
indices = np.random.permutation(len(X_private_norm))
partition_size = len(X_private_norm) // num_teachers

teacher_partitions = []
for i in range(num_teachers):
    start_idx = i * partition_size
    if i == num_teachers - 1:
        partition_indices = indices[start_idx:]
    else:
        partition_indices = indices[start_idx : start_idx + partition_size]
    teacher_partitions.append(partition_indices)

teachers = []
holdout_loader = create_dataloader(X_holdout_norm, y_holdout, 128, shuffle=False)

for i in tqdm(range(num_teachers), desc="Training teachers"):
    partition_idx = teacher_partitions[i]
    X_teacher = X_private_norm[partition_idx]
    y_teacher = y_private[partition_idx]

    teacher = MLP(
        input_size=DATASET_CONFIG["num_features"],
        hidden_layers=TEACHER_CONFIG["hidden_layers"],
        num_classes=DATASET_CONFIG["num_classes"],
        dropout=TEACHER_CONFIG["dropout"],
    )

    train_loader_t = create_dataloader(X_teacher, y_teacher, TEACHER_CONFIG["batch_size"])
    train_model(
        teacher,
        train_loader_t,
        holdout_loader,
        device=DEVICE,
        epochs=TEACHER_CONFIG["epochs"],
        learning_rate=TEACHER_CONFIG["learning_rate"],
    )
    teachers.append(teacher)

print(f"\nTeacher ensemble trained: {len(teachers)} teachers")
print("\n" + "=" * 60)
print("PHASE 3: PATE Aggregation")
print("=" * 60)

num_queries = min(AGGREGATION_CONFIG["num_student_queries"], len(X_public_norm))
query_indices = np.random.choice(len(X_public_norm), num_queries, replace=False)
X_query = X_public_norm[query_indices]
y_query_true = y_public[query_indices]
votes = get_teacher_votes(teachers, X_query, DEVICE)
noise_scale = AGGREGATION_CONFIG["noise_scale"]
threshold = AGGREGATION_CONFIG["confident_threshold"]
aggregation_output = pate_core_aggregation(votes, noise_scale, threshold)
if aggregation_output is None:
    raise NotImplementedError(
        "Implement `pate_core_aggregation` with noisy/confident aggregation "
        "to produce student training labels."
    )

student_labels, confident_mask = aggregation_output

X_student = X_query[confident_mask]
y_student = student_labels[confident_mask]
y_student_true = y_query_true[confident_mask]

label_accuracy = (y_student == y_student_true).mean() * 100
acceptance_rate = confident_mask.mean() * 100
print(f"Accepted samples: {len(y_student)} ({acceptance_rate:.1f}%)")
print(f"Label accuracy: {label_accuracy:.2f}%")

print("\n" + "=" * 60)
print("PHASE 4: Training Student Model")
print("=" * 60)

student_model = MLP(
    input_size=DATASET_CONFIG["num_features"],
    hidden_layers=STUDENT_CONFIG["hidden_layers"],
    num_classes=DATASET_CONFIG["num_classes"],
    dropout=STUDENT_CONFIG["dropout"],
)

student_train_loader = create_dataloader(X_student, y_student, STUDENT_CONFIG["batch_size"])
train_model(
    student_model,
    student_train_loader,
    holdout_loader,
    device=DEVICE,
    epochs=STUDENT_CONFIG["epochs"],
    learning_rate=STUDENT_CONFIG["learning_rate"],
)

student_test_acc = evaluate_accuracy(student_model, holdout_loader, DEVICE)
model_path = os.path.join(MODELS_DIR, "pate_student.safetensors")
save_file(student_model.state_dict(), model_path)

print("\n" + "=" * 60)
print("VALIDATION CHECK")
print("=" * 60)

member_idx = np.random.choice(len(X_private_norm), 2000, replace=False)
nonmember_idx = np.random.choice(len(X_holdout_norm), 2000, replace=False)
mia_adv = compute_mia_advantage_quick(
    student_model, X_private_norm[member_idx], X_holdout_norm[nonmember_idx], DEVICE
)

accuracy_pass = student_test_acc >= 80.0
mia_pass = mia_adv <= 0.03

print(f"Accuracy >= 80%:       {student_test_acc:.2f}% {'PASS' if accuracy_pass else 'FAIL'}")
print(f"MIA advantage <= 3%:   {mia_adv * 100:.2f}% {'PASS' if mia_pass else 'FAIL'}")
print(f"Saved model: {model_path}")

base_url = os.environ.get("BASE_URL")
if base_url:
    with open(model_path, "rb") as f:
        response = requests.post(
            f"{base_url}/validate",
            files={"model": ("pate_student.safetensors", f, "application/octet-stream")},
            timeout=120,
        )
    print(response.text)
else:
    print("Set BASE_URL to submit automatically.")
