# GoodWords Challenge — Scaffold Setup

Black-box Naive Bayes evasion via append-only word augmentation. Files live in this directory.

## 1) Enter the challenge directory

Why: Keep the entrypoint and setup doc together.

```bash
cd "$REPO_ROOT/08_AI_Evasion_Foundations"
```

## 2) Create and activate a virtual environment (recommended)

Why: Isolated deps on academy hosts where system site-packages may be read-only.

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

## 3) Install dependencies

Why: `requests` for the API client; `numpy` for the reproducibility seed used in the scaffold.

```bash
pip install requests numpy
```

## 4) Point at the hosted challenge instance

Why: All endpoints (`/health`, `/challenge`, `/predict`, `/submit`) share one base URL.

Replace placeholders with the instance IP and port from your lab/spawn panel, then export:

```bash
export BASE_URL="http://INSTANCE_IP:PORT"
```

Health check (expect `{"status":"healthy"}` or similar):

```bash
curl -s "${BASE_URL}/health"
```

Challenge parameters (note `base_message`, `max_added_words`, `target_label`):

```bash
curl -s "${BASE_URL}/challenge" | jq
```

Baseline prediction on the base message (expect high `spam_probability`):

```bash
curl -s -X POST "${BASE_URL}/predict" \
  -H 'content-type: application/json' \
  -d "{\"text\": $(curl -s "${BASE_URL}/challenge" | jq -r '.base_message | @json')}"
```

## 5) Run the scaffold

Why: Confirm imports, challenge fetch, vocabulary, and orchestration wire up; execution should stop at the first attack stub.

```bash
python3 blackbox_challenge.py
```

Expected behavior now:
- Base message and word budget print.
- Execution stops when `estimate_word_impacts` is reached (stub) or immediately after when its return value is consumed.

## 6) After implementing stubs

Why: `/submit` validates append-only constraints and word budget, then returns the flag when the model labels the augmented text as ham.

Re-run:

```bash
python3 blackbox_challenge.py
```

On success, stdout should include JSON with `result`, `details`, and `flag`.

Manual submit (optional sanity check):

```bash
curl -s -X POST "${BASE_URL}/submit" \
  -H 'content-type: application/json' \
  -d '{"augmented_text": "<your augmented message>"}' | jq
```

## 7) Local checks before relying on the flag

- `BASE_URL` is set and reachable (`/health` returns healthy).
- Augmented text equals the base message plus only appended tokens (no edits to the original text).
- Added token count stays within `max_added_words` from `/challenge`.
- Final `/predict` label is `ham` before `/submit`.
