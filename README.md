# AI Red Teamer — HTB Academy Notebooks & Solutions

Personal working notebooks, attack implementations, and challenge solvers for the
[Hack The Box Academy **AI Red Teamer** job-role path](https://academy.hackthebox.com/path/preview/ai-red-teamer)
(built in collaboration with Google — 12 modules, 230 sections, overall **Hard**).

Each top-level directory maps to an official module and is **numbered to match the
official path order**. Modules with no directory here (**01 · Fundamentals of AI**,
**04 · Prompt Injection Attacks**, **05 · LLM Output Attacks**) are not missing work —
their exercises required **no code** and were completed directly on the HTB platform
(e.g. interacting with a web chat to coax the target LLM into a given response). Every
module that involved code to solve is included.

> [!WARNING]
> **Educational use only.** This repository documents my own learning and contains
> spoilers/solutions for HTB Academy exercises. Please attempt the modules yourself
> first, and respect the [HTB Academy Terms of Service](https://www.hackthebox.com/tos).
> All attack code targets deliberately vulnerable lab instances only.

---

## Path modules

| # | Official module | Tier | Sections | In this repo |
|---|-----------------|------|:--------:|:------------:|
| 01 | Fundamentals of AI | Medium | 24 | on-platform (no code) |
| 02 | Applications of AI in InfoSec | Medium | 25 | ✅ [`02_Applications_of_AI_in_InfoSec/`](02_Applications_of_AI_in_InfoSec) |
| 03 | Introduction to Red Teaming AI | Medium | 11 | ✅ [`03_Introduction_to_Red_Teaming_AI/`](03_Introduction_to_Red_Teaming_AI) |
| 04 | Prompt Injection Attacks | Medium | 12 | on-platform (no code) |
| 05 | LLM Output Attacks | Medium | 14 | on-platform (no code) |
| 06 | AI Data Attacks | Hard | 25 | ✅ [`06_AI_Data_Attacks/`](06_AI_Data_Attacks) |
| 07 | Attacking AI - Application and System | Medium | 14 | ✅ [`07_Attacking_AI_Application_and_System/`](07_Attacking_AI_Application_and_System) |
| 08 | AI Evasion - Foundations | Medium | 12 | ✅ [`08_AI_Evasion_Foundations/`](08_AI_Evasion_Foundations) |
| 09 | AI Evasion - First-Order Attacks | Hard | 23 | ✅ [`09_AI_Evasion_First-Order_Attacks/`](09_AI_Evasion_First-Order_Attacks) |
| 10 | AI Evasion - Sparsity Attacks | Hard | 28 | ✅ [`10_AI_Evasion_Sparsity_Attacks/`](10_AI_Evasion_Sparsity_Attacks) |
| 11 | AI Privacy | Medium | 21 | ✅ [`11_AI_Privacy/`](11_AI_Privacy) |
| 12 | AI Defense | Medium | 21 | ✅ [`12_AI_Defense/`](12_AI_Defense) |

---

## What's in each module

**02 · Applications of AI in InfoSec** — foundational applied-ML notebooks.
- `1_introduction.ipynb` — Python/ML libraries primer
- `2_spam_classification.ipynb` — Naive Bayes spam filter (SMS Spam Collection)
- `3_network_anomaly_detection.ipynb` — anomaly detection on the KDD dataset
- `4_malware_classification.ipynb` — CNN image classifier on the MalImg dataset

**03 · Introduction to Red Teaming AI** — `redteam_code/` end-to-end data-poisoning demo
(`main.py` trains on `poison.csv`, evaluates on `test.csv`) plus reference material.

**06 · AI Data Attacks** — data-poisoning attack notebooks:
- `label_flipping/` — untargeted and targeted label-flipping (shared `label_flipping_dataset.npz`)
- `clean_label/` — clean-label poisoning (`clean_label_eval_dataset.npz`)
- `trojan/` — MNIST backdoor / trojan trigger
- `steganography/` — pickle & tensor-LSB steganography payload embedding
- `bin_ops.py` — bit-level helpers

**07 · Attacking AI - Application and System** — model reverse engineering, sponge /
Denial-of-ML-Service, and MCP (Model Context Protocol) attacks:
- `attacking_the_application.py`, `sponge_attack.py`, `mcp_client.py`, `mcp_server.py`
- `exercises/` — model-extraction (`rev_eng_model.py`) and MCP exploitation (`mcp_challenge.py`)

**08 · AI Evasion - Foundations** — the GoodWords spam-filter evasion attack, white-box
(`goodwords_attack.ipynb`) and black-box (`black_box_goodwords_attack.ipynb`), plus the
`blackbox_challenge.py` solver.

**09 · AI Evasion - First-Order Attacks** — gradient-based evasion: FGSM (`fgsm.ipynb`,
`fgsm_challenge.py`) and DeepFool (`deepfool.ipynb`, `deepfool.py`).

**10 · AI Evasion - Sparsity Attacks** — L0 / sparse perturbation attacks:
- `jsma.ipynb` / `jsma_attack/` — Jacobian-based Saliency Map Attack
- `elasticnet_attack/` — ElasticNet (EAD) attack implementation

**11 · AI Privacy** — membership inference and differential privacy:
- `shadow_model_attack.py` — shadow-model membership inference
- `dp_sgd.ipynb` + `dp_sgd_svhn/` — differentially private SGD
- `pate.ipynb` + `pate_challenge/` — Private Aggregation of Teacher Ensembles

**12 · AI Defense** — `adversarial_training.ipynb` / `adversarial_tuning.py` (adversarial
robustness) and `llm_guardrails.py` (Pydantic-based LLM I/O guardrails).

---

## Setup

Most notebooks target Python 3.11 and share one environment. A helper script creates a
conda env, installs `requirements.txt`, and adds the HTB AI helper library:

```bash
./setup.sh          # creates/activates the `ai_red_teamer` conda env and installs deps
```

Or manually:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --upgrade git+https://github.com/PandaSt0rm/htb-ai-library
```

### Running the challenge solvers

Several modules include a `setup.md` with per-challenge steps. Live challenge endpoints
are **not** hardcoded — set the target of your spawned lab instance via env vars or the
`INSTANCE_IP:PORT` placeholders in the scripts, e.g.:

```bash
export REPO_ROOT="$(pwd)"
export CLASSIFIER_URL="http://<INSTANCE_IP>:<PORT>/"   # module 07
export BASE_URL="http://<INSTANCE_IP>:<PORT>"          # module 08 GoodWords challenge
```

### Data

Notebooks download their own datasets on first run (MNIST, SVHN, SMS Spam, KDD, MalImg,
etc.) — these are `.gitignore`d. Only the small, non-regenerable **challenge input
datasets** (the `.npz` evaluation sets and the poisoning `.csv` files) are committed so
the exercises are runnable out of the box. Generated model weights (`*.pth`, `*.joblib`)
and plot outputs are ignored.
