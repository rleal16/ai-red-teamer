import os
import random
import pandas as pd
import requests
import json


N_SAMPLES = 100

MIN_FLIPPER_LENGTH = 150
MAX_FLIPPER_LENGTH = 250

MIN_BODY_MASS = 2500
MAX_BODY_MASS = 6500

# Set to your spawned lab instance, e.g. CLASSIFIER_URL=http://INSTANCE_IP:PORT/
CLASSIFIER_URL = os.environ.get("CLASSIFIER_URL", "http://INSTANCE_IP:PORT/")


samples = {
    "Flipper Length (mm)": [],
    "Body Mass (g)": []
}

for i in range(N_SAMPLES):
    samples["Flipper Length (mm)"].append(random.uniform(MIN_FLIPPER_LENGTH, MAX_FLIPPER_LENGTH))
    samples["Body Mass (g)"].append(random.uniform(MIN_BODY_MASS, MAX_BODY_MASS))

samples_df = pd.DataFrame(samples)
print(samples_df.head())

# send the generated input data to the target model and collect the results

predictions = {"species": []}
for i in range(N_SAMPLES):
    sample = {
        "flipper_length": samples["Flipper Length (mm)"][i],
        "body_mass": samples["Body Mass (g)"][i]
    }

    prediction = json.loads(
        requests.get(CLASSIFIER_URL, params=sample).text
    ).get("result")
    predictions["species"].append(prediction)

predictions_df = pd.DataFrame(predictions)
print(predictions_df.head())

# train the surrogate model -- Logistic Regression in this case -- with the data obtained above from the predicitons of the other models
# choosing (and guessing) the exact model equal to the target's is often impossibel, however this is not necessary as long as the model is the right for the overall task

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

surrogate_model = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)
surrogate_model.fit(samples_df, predictions_df)

# save classifier to a file
joblib.dump(surrogate_model, 'surrogate.joblib')

with open('surrogate.joblib', 'rb') as f:
    file = f.read()

r = requests.post(CLASSIFIER_URL + '/model', files={'file':{'surrogate.joblib', file}})

print(json.loads(r.text))