import os
import random

import numpy as np
import requests

random.seed(1337)
np.random.seed(1337)

host = os.getenv("BASE_URL")
ch = requests.get(f"{host}/challenge", timeout=10).json()
base, budget = ch["base_message"], int(ch["max_added_words"])


def predict(host, text):
    r = requests.post(f"{host}/predict", json={"text": text}, timeout=15)
    data = r.json()
    return data["label"], float(data["spam_probability"])


vocab = list(
    dict.fromkeys(
        [
            "hello",
            "hi",
            "dear",
            "please",
            "thanks",
            "thankyou",
            "regards",
            "sincerely",
            "kind",
            "wishes",
            "best",
            "appreciate",
            "welcome",
            "friend",
            "family",
            "meeting",
            "tomorrow",
            "today",
            "later",
            "morning",
            "night",
            "home",
            "work",
            "office",
            "schedule",
            "confirm",
            "call",
            "reply",
            "message",
            "note",
            "update",
            "happy",
            "birthday",
            "congratulations",
            "joy",
            "peace",
            "smile",
            "care",
            "support",
            "help",
            "good",
            "great",
            "wonderful",
            "awesome",
            "nice",
            "cool",
            "fine",
            "okay",
            "lunch",
            "dinner",
            "coffee",
            "weekend",
            "party",
            "invite",
            "visit",
            "enjoy",
            "true",
            "honest",
            "trust",
            "safe",
            "project",
            "team",
        ]
    )
)


def estimate_word_impacts(*args):
    _, base_prob = predict(host, base)
    deltas = []
    for w in vocab:
        mod_msg = f"{base} {w}"
        _, prob = predict(host, mod_msg)
        delta = base_prob - prob
        deltas.append((w, delta))
    
    deltas.sort(key=lambda x: x[1], reverse=True)

    return deltas


def greedy_compose(*args, **kwargs):
    host, base, top_words, budget = args
    target_label = kwargs["target_label"]
    label = None
    words = iter(top_words)
    used = []
    prob = 1
    while budget > 0 and label != target_label:
        try:
            w = next(words)
        except StopIteration:
            break
        base += " " + w    
        label, prob = predict(host, base)
        used.append(w)
        budget -= 1
    
    return base, len(used), prob
        
    
    


if __name__ == "__main__":
    print(f"Base message: '{base}'\nBudget: {budget}")

    impacts = estimate_word_impacts(host, base, vocab)
    top_words = [w for w, delta in impacts if delta > 0]

    final_msg, used, final_prob = greedy_compose(
        host, base, top_words, budget, target_label="ham"
    )
    final_label, _ = predict(host, final_msg)

    print(f"\nFinal message: '{final_msg}'")
    print(f"Words used: {used}")
    print(f"Final classification: {final_label} (spam prob: {final_prob:.4f})")

    submit_resp = requests.post(
        f"{host}/submit", json={"augmented_text": final_msg}, timeout=15
    ).json()
    print(submit_resp)
