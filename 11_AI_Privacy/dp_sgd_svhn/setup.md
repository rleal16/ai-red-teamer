# DP-SGD Privacy Challenge (SVHN) — local scaffold

## Layout

```text
11_AI_Privacy/dp_sgd_svhn/
  setup.md          # this file
  train.py          # local training + local checks; DP-SGD block is stubbed
  data/             # created on first SVHN download (under this directory if cwd is here)
  models/           # created by train.py; holds dp_model.safetensors
```

Run all commands from `dp_sgd_svhn/` so paths match the script (`data/`, `models/`).

## Virtual environment

Why: isolate `torch`, `torchvision`, `opacus`, and pinned deps.

From the repository root:

```bash
cd 11_AI_Privacy/dp_sgd_svhn
python3 -m venv venv
source venv/bin/activate
```

Windows (if needed):

```powershell
python -m venv venv
.\venv\Scripts\activate
```

## Dependencies

Why: SVHN + DP training (Opacus) + export + optional remote validate.

```bash
pip install numpy torch torchvision requests safetensors packaging tqdm scipy opacus
```

Health check (imports + CPU/CUDA):

```bash
python -c "import torch, torchvision, opacus; print('ok', torch.__version__)"
```

## HTB instance

Why: the challenge API loads data server-side; you only upload weights.

Current spawn (replace when the instance rotates):

```bash
export BASE_URL="http://INSTANCE_IP:PORT"
```

Smoke the service:

```bash
curl -s "$BASE_URL/health" | jq
curl -s "$BASE_URL/" | jq
```

## Train locally

Why: builds `models/dp_model.safetensors` and runs the same accuracy / MIA-style checks as in the reference flow (after you implement the stub).

```bash
source venv/bin/activate
python train.py
```

## Submit to `/validate`

Why: returns the flag when `passed` is true.

```bash
curl -s -X POST "$BASE_URL/validate" \
  -F "model=@models/dp_model.safetensors" | jq
```

Equivalent from Python (set `BASE_URL` first):

```bash
python -c "
import os, requests
base = os.environ['BASE_URL']
with open('models/dp_model.safetensors', 'rb') as f:
    r = requests.post(f'{base}/validate', files={'model': ('dp_model.safetensors', f, 'application/octet-stream')})
print(r.json())
"
```

## Notes

- SVHN is downloaded into `./data` on first run (`download=True` on the first loader call in `train.py`).
- Opacus expects an Opacus-compatible module; the reference uses `ModuleValidator.fix` before attaching `PrivacyEngine`.
- After DP wrapping, weights live on `model._module` for `safetensors` export.
