# DeepFool Challenge — Scaffold Setup

Targeted DeepFool-style adversarial crafting against a hosted MNIST classifier under an ℓ₂ distance constraint. Files live in this directory.

## 1) Enter the challenge directory

Why: Keep weights, the entrypoint, and this setup doc together.

```bash
cd "$REPO_ROOT/09_AI_Evasion_First-Order_Attacks"
```

## 2) Create and activate a virtual environment (recommended)

Why: Isolated deps on academy hosts where system site-packages may be read-only.

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

## 3) Install dependencies

Why: PyTorch for local gradient computation; `requests`/`Pillow`/`numpy` for API I/O and image encoding.

On a local machine:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install requests pillow numpy
```

On HTB academy workstations with tight disk space, reclaim space first if `pip install torch` fails, then install the CPU wheel:

```bash
sudo apt autoremove
sudo apt clean
sudo rm -rf /tmp/*
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install requests pillow numpy
```

## 4) Point at the hosted challenge instance

Why: `/health`, `/challenge`, `/predict`, `/weights`, and `/submit` share one base URL.

Replace placeholders with the instance IP and port from your lab/spawn panel:

```bash
export BASE_URL="http://INSTANCE_IP:PORT"
```

Health check (expect `status`, `l2_threshold`, `index`, and `target`):

```bash
curl -s "${BASE_URL}/health" | jq
```

Challenge payload (note `label`, `target`, `l2_threshold`, and baseline `image_b64` in `[0,1]` pixel space after decode):

```bash
curl -s "${BASE_URL}/challenge" | jq
```

Weights download smoke test (binary PyTorch state dict):

```bash
curl -s -o deepfool_weights.pth "${BASE_URL}/weights"
python3 -c "import torch; torch.load('deepfool_weights.pth', map_location='cpu'); print('weights ok')"
```

Local predict sanity check on the clean image:

```bash
curl -s -X POST "${BASE_URL}/predict" \
  -H 'content-type: application/json' \
  -d "{\"image_b64\": $(curl -s "${BASE_URL}/challenge" | jq -r '.image_b64')}" | jq
```

## 5) Run the scaffold

Why: Confirm imports, challenge fetch, model load, and orchestration wire up; execution should stop at the DeepFool stub.

```bash
python3 deepfool.py --host "${BASE_URL}"
```

Expected behavior now:
- Challenge connects and weights download to `solver/deepfool_weights.pth` if missing.
- Local clean prediction prints (should match `/predict` on the baseline).
- Execution stops when `deepfool_targeted` is reached (stub returns `None`) or on the immediate use of its return value.

## 6) After implementing the stub

Why: `/submit` checks `[0,1]` bounds, ℓ₂ distance ≤ `l2_threshold`, and predicted class equals `target` before returning the flag.

Re-run:

```bash
python3 deepfool.py --host "${BASE_URL}"
```

On success, stdout should include adversarial `l2`/`pred` JSON and `Flag: HTB{...}`.

Manual predict/submit checks (optional):

```bash
# Expect 400 — clean image predicts the true label, not the target
curl -s -X POST "${BASE_URL}/submit" \
  -H 'content-type: application/json' \
  -d "{\"image_b64\": $(curl -s "${BASE_URL}/challenge" | jq -r '.image_b64')}"
```

## 7) Local checks before relying on the flag

- Images stay in `[0,1]` pixel space for API transport (model normalizes internally via MNIST mean/std).
- ℓ₂ is measured in `[0,1]` space against the baseline from `/challenge`, not normalized tensors.
- PNG quantization can nudge predictions and distance; the scaffold retries several overshoot values after you implement the attack.
- Local `SimpleClassifier` + downloaded weights should match `/predict` on the clean image before submitting.
- Success requires **targeted** misclassification (`pred == target`), not merely any wrong class.
