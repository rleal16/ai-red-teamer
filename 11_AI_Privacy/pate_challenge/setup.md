# PATE Challenge Scaffold Setup

This scaffold is for the EMNIST Letters PATE challenge under `12_AI_Defense/pate_challenge`.

## 1) Enter the challenge directory

Why: Keep dependencies and outputs isolated to this challenge.

```bash
cd "$REPO_ROOT/11_AI_Privacy/pate_challenge"
```

## 2) Create and activate a virtual environment

Why: Reproducible Python environment for training and validation.

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

## 3) Install required dependencies

Why: Required for EMNIST loading, model training, secure model export, and validator submission.

```bash
pip3 install numpy torch torchvision requests safetensors packaging tqdm scipy scikit-learn
```

## 4) Configure target validator URL

Why: `pate.py` reads this value for `/health` and `/validate` checks.

```bash
export BASE_URL="http://INSTANCE_IP:PORT"
```

Optional quick connectivity checks:

```bash
curl -s "$BASE_URL/" | jq
curl -s "$BASE_URL/health" | jq
```

## 5) Run the scaffold

Why: Verify imports, data loading path, and boilerplate execution; the run is expected to stop at the core PATE stub.

```bash
python3 pate.py
```

Expected behavior now:
- Dataset download/load starts successfully.
- Boilerplate runs.
- Execution halts with `NotImplementedError` at the PATE aggregation stub.

## 6) Save and submit model (after implementing stub)

The script writes:
- `models/pate_student.safetensors`

Submit manually:

```bash
python3 - <<'PY'
import os
import requests

base_url = os.environ["BASE_URL"]
with open("models/pate_student.safetensors", "rb") as f:
    response = requests.post(
        f"{base_url}/validate",
        files={"model": ("pate_student.safetensors", f, "application/octet-stream")},
        timeout=120,
    )
print(response.text)
PY
```

## 7) Local health checks

Before submission, confirm:
- Student architecture exactly matches challenge MLP shape and forward path.
- EMNIST labels are converted from `1-26` to `0-25`.
- StandardScaler is fit on train split and applied to test/public splits.
- Output file exists at `models/pate_student.safetensors`.
