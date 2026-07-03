import json
import os
import random

import joblib
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_SAMPLES = 100

MIN_FLIPPER_LENGTH = 150
MAX_FLIPPER_LENGTH = 250

MIN_BODY_MASS = 2500
MAX_BODY_MASS = 6500

CLASSIFIER_URL = os.environ.get("CLASSIFIER_URL", "http://INSTANCE_IP:PORT/")


samples = {
    "Flipper Length (mm)": [],
    "Body Mass (g)": [],
}

for i in range(N_SAMPLES):
    samples["Flipper Length (mm)"].append(
        random.uniform(MIN_FLIPPER_LENGTH, MAX_FLIPPER_LENGTH)
    )
    samples["Body Mass (g)"].append(random.uniform(MIN_BODY_MASS, MAX_BODY_MASS))

samples_df = pd.DataFrame(samples)
print(samples_df.head())


def collect_oracle_labels():
    preds = {"species": []}
    for i in range(N_SAMPLES):
        flipper_len = samples["Flipper Length (mm)"][i]
        body_mass = samples["Body Mass (g)"][i]
        req_str = f"{CLASSIFIER_URL}?flipper_length={flipper_len}&body_mass={body_mass}"
        r = requests.get(req_str)
        r.raise_for_status()
        label = r.json()["result"]
        preds["species"].append(label)
    preds_df = pd.DataFrame(preds)
    return preds_df



def train_stolen_model(samples: pd.DataFrame, labels: pd.DataFrame):
    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(samples, labels.values.ravel())

    return model

predictions_df = collect_oracle_labels()
print(predictions_df.head())

stolen_model = train_stolen_model(samples_df, predictions_df)

joblib.dump(stolen_model, "student.joblib")

with open("student.joblib", "rb") as f:
    file = f.read()

r = requests.post(
    CLASSIFIER_URL + "model",
    files={"file": ("student.joblib", file)},
)

print("\n===========\n")
print(json.loads(r.text))
