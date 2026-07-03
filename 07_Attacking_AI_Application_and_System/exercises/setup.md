# Model Reverse Engineering — Q1 Scaffold Setup

Scaffold for the hosted penguin classifier extraction challenge. Files live in this directory.

## 1) Enter the exercise directory

Why: Keep `student.joblib` and script outputs next to the entrypoint.

```bash
cd "$REPO_ROOT/07_Attacking_AI_Application_and_System/exercises"
```

## 2) Create and activate a virtual environment (recommended)

Why: Isolated deps on academy hosts where system site-packages may be read-only.

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

## 3) Install dependencies

Why: Course module lists these packages; the script also needs `requests` and `joblib` for querying and model export.

```bash
pip install nltk pandas scikit-learn requests joblib
```

## 4) Point at the hosted classifier

Why: Probe queries and model submission both use the same base URL.

Replace placeholders with the instance IP and port from your lab/spawn panel, then export:

```bash
export CLASSIFIER_URL="http://INSTANCE_IP:PORT/"
```

Quick health check (expect JSON with a species label or similar):

```bash
curl -s "${CLASSIFIER_URL}?flipper_length=200&body_mass=4000"
```

## 5) Run the scaffold

Why: Confirm imports, synthetic sample generation, and submission boilerplate wire up; execution should stop at the first attack stub.

```bash
python3 rev_eng_model.py
```

Expected behavior now:
- Sample DataFrame prints (`head()`).
- Execution stops when the first stub is reached (oracle label collection) or on the immediate use of its return value.

## 6) After implementing stubs

Why: The validator scores your uploaded surrogate against the hidden model; the module expects ≥80% accuracy (reference runs often land ~98%).

Re-run:

```bash
python3 rev_eng_model.py
```

On success, stdout should include JSON with `accuracy` and `flag` after the `===========` banner.

Artifacts:
- `student.joblib` — serialized surrogate written before POST to `/model`

## 7) Local checks before relying on the flag

- `CLASSIFIER_URL` ends with `/` (script appends `model` for upload).
- GET query params use `flipper_length` and `body_mass` (snake_case), not the DataFrame column names.
- Surrogate is fit on the same feature matrix used for probing (flipper length + body mass).
- Upload uses multipart field name `file` and filename `student.joblib`.
