import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from htb_ai_library import (
    set_reproducibility, use_htb_style,
    MLP, AttackModel,
    load_adult_census,
    train_fixed_epochs, train_with_early_stopping, evaluate_model,
    get_model_predictions, prepare_attack_data, create_dataloader,
    plot_training_history, plot_overfitting_gap, plot_confidence_distributions,
    plot_shadow_confidence_distributions, plot_attack_roc_curve, plot_precision_recall_curve,
    plot_attack_accuracy_comparison, analyze_attack_decision_boundary, plot_decision_boundary,
)

RANDOM_SEED = 1337
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_reproducibility(RANDOM_SEED)
use_htb_style()

# Output directories for saving models and figures
OUTPUT_DIR = "output"
MODEL_DIR = f"{OUTPUT_DIR}/models"
FIGS_DIR = "figs"
FIG_PREFIX = "Introduction_"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

DATASET_CONFIG = {
    "num_classes": 2,
}

# config done to maximize the gap of how models treat seen (training) data vs unseen data
# i.e., to maximize overfitting
TARGET_MODEL_CONFIG = {
    "hidden_layers": [256, 128],
    "dropout": 0.0,  # No dropout to maximize overfitting
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
}

SHADOW_MODEL_CONFIG = {
    "num_shadow_models": 5,
    "hidden_layers": [128, 64],
    "dropout": 0.3,
    "epochs": 100,
    "batch_size": 64,
    "learning_rate": 0.001,
    "early_stopping_patience": 10,
    "shadow_data_size": 0.5,
}

ATTACK_MODEL_CONFIG = {
    "hidden_layers": [64, 32],
    "dropout": 0.2,
    "epochs": 100,
    "batch_size": 128,
    "learning_rate": 0.001,
    "early_stopping_patience": 15,
}

# Total Dataset (48,842 samples)
# ├── Target Training (24,421) → Members we try to identify
# └── Holdout (24,421)
#     ├── Shadow Training (12,210) → Train shadow models
#     └── Attack Evaluation (12,210) → Non-members for final testing


print("Loading Adult Census dataset...")
X_target, y_target, X_shadow, y_shadow, X_attack_eval, y_attack_eval, num_features = load_adult_census(
    random_state=RANDOM_SEED
)

print(f"Dataset loaded: {num_features} features")
print(f"  Target training (members): {len(X_target)} samples")
print(f"  Shadow training: {len(X_shadow)} samples")
print(f"  Attack evaluation (non-members): {len(X_attack_eval)} samples")

print("\n" + "=" * 60)
print("Training Target Model (to overfit deliberately)")
print("=" * 60)

scaler = StandardScaler()
X_target_norm = scaler.fit_transform(X_target)
X_attack_eval_norm = scaler.transform(X_attack_eval)

# create DataLoaders without a validation split. We deliberately omit validation because we want maximum overfitting
train_loader = create_dataloader(X_target_norm, y_target, TARGET_MODEL_CONFIG['batch_size'])
test_loader = create_dataloader(X_attack_eval_norm, y_attack_eval,
                                TARGET_MODEL_CONFIG['batch_size'], shuffle=False)


# initialize the target model without dropout
target_model = MLP(
    input_size=num_features,
    hidden_layers=TARGET_MODEL_CONFIG['hidden_layers'],
    num_classes=DATASET_CONFIG['num_classes'],
    dropout=TARGET_MODEL_CONFIG['dropout']
)

print(f"Architecture: {num_features} -> {TARGET_MODEL_CONFIG['hidden_layers']} -> 2")
print(f"Training for {TARGET_MODEL_CONFIG['epochs']} epochs (no early stopping)")

# we intentionally train for the full 100 epochs to maximize overfitting
history = train_fixed_epochs(
    target_model, train_loader, test_loader,
    device=DEVICE,
    epochs=TARGET_MODEL_CONFIG['epochs'],
    learning_rate=TARGET_MODEL_CONFIG['learning_rate']
)


train_acc, _, _ = evaluate_model(target_model, train_loader, DEVICE)
test_acc, _, _ = evaluate_model(target_model, test_loader, DEVICE)

print(f"\nTarget Model Performance:")
print(f"  Training Accuracy: {train_acc:.4f}")
print(f"  Test Accuracy:     {test_acc:.4f}")
print(f"  Overfitting Gap:   {train_acc - test_acc:.4f}")

plot_overfitting_gap(train_acc, test_acc,
                     save_path=os.path.join(FIGS_DIR, f"{FIG_PREFIX}overfitting_gap.png"))


# craete shadow model data splits
print("\n" + "=" * 60)
print("Training Shadow Models")
print("=" * 60)

shadow_splits = []
for i in range(SHADOW_MODEL_CONFIG['num_shadow_models']):
    seed = RANDOM_SEED + i
    X_train_s, X_out_s, y_train_s, y_out_s = train_test_split(
        X_shadow, y_shadow, train_size=SHADOW_MODEL_CONFIG['shadow_data_size'],
        random_state=seed, stratify=y_shadow
    )
    shadow_splits.append((X_train_s, X_out_s, y_train_s, y_out_s))

print(f"\nCreated {len(shadow_splits)} shadow model data splits")
print(f"Samples per shadow model: ~{len(shadow_splits[0][0])} in, ~{len(shadow_splits[0][1])} out")

all_attack_X = []
all_attack_y = []
all_preds_in = []
all_preds_out = []

for i, (X_train_s, X_out_s, y_train_s, y_out_s) in enumerate(shadow_splits):
    print(f"\nTraining Shadow Model {i+1}/{SHADOW_MODEL_CONFIG['num_shadow_models']}")

    X_train_s_norm = scaler.transform(X_train_s)
    X_out_s_norm = scaler.transform(X_out_s)

    # Create validation split for early stopping
    X_tr_s, X_val_s, y_tr_s, y_val_s = train_test_split(
        X_train_s_norm, y_train_s, test_size=0.2,
        random_state=RANDOM_SEED + i, stratify=y_train_s
    )
    train_loader_s = create_dataloader(X_tr_s, y_tr_s, SHADOW_MODEL_CONFIG['batch_size'])
    val_loader_s = create_dataloader(X_val_s, y_val_s, SHADOW_MODEL_CONFIG['batch_size'], shuffle=False)

    # Initialize and train shadow model
    shadow_model = MLP(
        input_size=num_features,
        hidden_layers=SHADOW_MODEL_CONFIG['hidden_layers'],
        num_classes=DATASET_CONFIG['num_classes'],
        dropout=SHADOW_MODEL_CONFIG['dropout']
    )
    train_with_early_stopping(
        shadow_model, train_loader_s, val_loader_s,
        device=DEVICE,
        epochs=SHADOW_MODEL_CONFIG['epochs'],
        learning_rate=SHADOW_MODEL_CONFIG['learning_rate'],
        patience=SHADOW_MODEL_CONFIG['early_stopping_patience'],
        verbose=False
    )

    # Collect predictions on members and non-members
    preds_in = get_model_predictions(shadow_model, X_train_s_norm, DEVICE)
    preds_out = get_model_predictions(shadow_model, X_out_s_norm, DEVICE)

    # Transform to attack features and accumulate
    attack_X_s, attack_y_s = prepare_attack_data(preds_in, preds_out, y_train_s, y_out_s)
    all_attack_X.append(attack_X_s)
    all_attack_y.append(attack_y_s)
    all_preds_in.append(preds_in)
    all_preds_out.append(preds_out)

    # Verify overfitting gap exists
    full_train_loader_s = create_dataloader(X_train_s_norm, y_train_s,
                                            SHADOW_MODEL_CONFIG['batch_size'], shuffle=False)
    full_out_loader_s = create_dataloader(X_out_s_norm, y_out_s,
                                          SHADOW_MODEL_CONFIG['batch_size'], shuffle=False)
    train_acc_s, _, _ = evaluate_model(shadow_model, full_train_loader_s, DEVICE)
    out_acc_s, _, _ = evaluate_model(shadow_model, full_out_loader_s, DEVICE)
    print(f"  Shadow {i+1} - Train Acc: {train_acc_s:.4f}, Out Acc: {out_acc_s:.4f}")


    attack_X = np.concatenate(all_attack_X, axis=0)
    attack_y = np.concatenate(all_attack_y, axis=0)

    print(f"\nTotal attack training samples: {len(attack_X)}")
    print(f"  Members: {np.sum(attack_y == 1)}")
    print(f"  Non-members: {np.sum(attack_y == 0)}")

    print(f"\nAttack feature dimensions: {attack_X.shape[1]}")
    print(f"Example member feature: {attack_X[0].round(3)}")
    print(f"Example non-member feature: {attack_X[len(attack_X)//2].round(3)}")

plot_shadow_confidence_distributions(
    all_preds_in, all_preds_out,
    save_path=os.path.join(FIGS_DIR, f"{FIG_PREFIX}shadow_confidence.png")
)

print("\n" + "=" * 60)
print("Training Attack Model")
print("=" * 60)

X_attack_train, X_attack_test, y_attack_train, y_attack_test = train_test_split(
    attack_X, attack_y, test_size=0.2, random_state=RANDOM_SEED, stratify=attack_y
)

print(f"\nAttack data split:")
print(f"  Training + Validation: {len(X_attack_train)} samples")
print(f"  Test: {len(X_attack_test)} samples")

X_attack_tr, X_attack_val, y_attack_tr, y_attack_val = train_test_split(
    X_attack_train, y_attack_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_attack_train
)

print(f"  Training: {len(X_attack_tr)} samples")
print(f"  Validation: {len(X_attack_val)} samples")

attack_train_loader = create_dataloader(X_attack_tr, y_attack_tr, ATTACK_MODEL_CONFIG['batch_size'])
attack_val_loader = create_dataloader(X_attack_val, y_attack_val, ATTACK_MODEL_CONFIG['batch_size'], shuffle=False)
attack_test_loader = create_dataloader(X_attack_test, y_attack_test, ATTACK_MODEL_CONFIG['batch_size'], shuffle=False)

print(f"\nDataLoaders created with batch size {ATTACK_MODEL_CONFIG['batch_size']}")

attack_input_size = attack_X.shape[1]
attack_model = AttackModel(
    input_size=attack_input_size,
    hidden_layers=ATTACK_MODEL_CONFIG['hidden_layers'],
    dropout=ATTACK_MODEL_CONFIG['dropout']
)

print(f"\nAttack model architecture: {attack_input_size} -> {ATTACK_MODEL_CONFIG['hidden_layers']} -> 2")
print(f"Dropout: {ATTACK_MODEL_CONFIG['dropout']}")

print("\nTraining attack model...")

history_attack = train_with_early_stopping(
    attack_model, attack_train_loader, attack_val_loader,
    device=DEVICE,
    epochs=ATTACK_MODEL_CONFIG['epochs'],
    learning_rate=ATTACK_MODEL_CONFIG['learning_rate'],
    patience=ATTACK_MODEL_CONFIG['early_stopping_patience']
)

plot_training_history(
    history_attack,
    "Attack Model Training",
    save_path=os.path.join(FIGS_DIR, f"{FIG_PREFIX}attack_training.png")
)

attack_test_acc, attack_test_predictions, attack_test_probs = evaluate_model(attack_model, attack_test_loader, DEVICE)

print(f"\nAttack Model Test Performance:")
print(f"  Accuracy: {attack_test_acc:.4f}")
print(f"  Samples: {len(attack_test_predictions)}")

print("\nDetailed Classification Report:")
print(classification_report(
    y_attack_test,
    attack_test_predictions,
    target_names=['Non-Member', 'Member'],
    digits=4
))

# Save the attack model
attack_model_path = os.path.join(MODEL_DIR, "attack_model.pt")
torch.save(attack_model.state_dict(), attack_model_path)
print(f"\nAttack model saved to {attack_model_path}")

boundary_analysis = analyze_attack_decision_boundary(attack_model, DEVICE)

print("\nDecision Boundary Analysis:")
for cls, data in boundary_analysis.items():
    threshold_idx = np.argmin(np.abs(data['membership_probs'] - 0.5))
    threshold_conf = data['confidences'][threshold_idx]
    print(f"  Class {cls}: Membership threshold at confidence ~{threshold_conf:.3f}")


# Executing the membership inference attack
print("\n" + "=" * 60)
print("Executing Membership Inference Attack")
print("=" * 60)

preds_members = get_model_predictions(target_model, X_target_norm, DEVICE)
preds_non_members = get_model_predictions(target_model, X_attack_eval_norm, DEVICE)

print(f"\nTarget model predictions collected:")
print(f"  Members: {len(preds_members)} samples")
print(f"  Non-members: {len(preds_non_members)} samples")


attack_X_members, attack_y_members = prepare_attack_data(
    preds_members, np.zeros((0, preds_members.shape[1])),
    y_target, np.array([], dtype=np.int64)
)

attack_X_non_members, attack_y_non_members = prepare_attack_data(
    np.zeros((0, preds_non_members.shape[1])), preds_non_members,
    np.array([], dtype=np.int64), y_attack_eval
)

print(f"\nAttack input prepared:")
print(f"  Member features: {attack_X_members.shape}")
print(f"  Non-member features: {attack_X_non_members.shape}")

attack_X_eval = np.concatenate([attack_X_members, attack_X_non_members], axis=0)
attack_y_eval = np.concatenate([attack_y_members, attack_y_non_members], axis=0)

print(f"\nTotal attack evaluation samples: {len(attack_X_eval)}")
print(f"  Members: {np.sum(attack_y_eval == 1)}")
print(f"  Non-members: {np.sum(attack_y_eval == 0)}")


attack_eval_loader = create_dataloader(attack_X_eval, attack_y_eval, ATTACK_MODEL_CONFIG['batch_size'], shuffle=False)

_, attack_predictions, attack_probs = evaluate_model(attack_model, attack_eval_loader, DEVICE)

membership_probs = attack_probs[:, 1]

print(f"\nAttack predictions generated")
print(f"  Mean membership probability: {membership_probs.mean():.4f}")

attack_accuracy = accuracy_score(attack_y_eval, attack_predictions)
attack_precision = precision_score(attack_y_eval, attack_predictions)
attack_recall = recall_score(attack_y_eval, attack_predictions)
attack_f1 = f1_score(attack_y_eval, attack_predictions)

print(f"\nMembership Inference Attack Results:")
print(f"  Attack Accuracy:  {attack_accuracy:.4f}")
print(f"  Attack Precision: {attack_precision:.4f}")
print(f"  Attack Recall:    {attack_recall:.4f}")
print(f"  Attack F1 Score:  {attack_f1:.4f}")


results = {
    'attack_accuracy': attack_accuracy,
    'attack_precision': attack_precision,
    'attack_recall': attack_recall,
    'attack_f1': attack_f1,
    'attack_y_true': attack_y_eval,
    'attack_y_pred': attack_predictions,
    'attack_probs': membership_probs,
    'confidence_members': np.max(preds_members, axis=1),
    'confidence_non_members': np.max(preds_non_members, axis=1),
}

print("\nResults stored for visualization")


print("\n" + "=" * 60)
print("Generating Visualizations")
print("=" * 60)

auc_score = plot_attack_roc_curve(
    results['attack_y_true'],
    results['attack_probs'],
    save_path=os.path.join(FIGS_DIR, f"{FIG_PREFIX}attack_roc.png")
)
results['attack_auc'] = auc_score

print(f"Attack AUC: {auc_score:.4f}")


plot_precision_recall_curve(
    results['attack_y_true'],
    results['attack_probs'],
    save_path=os.path.join(FIGS_DIR, f"{FIG_PREFIX}attack_pr.png")
)

plot_confidence_distributions(
    results['confidence_members'],
    results['confidence_non_members'],
    save_path=os.path.join(FIGS_DIR, f"{FIG_PREFIX}confidence_distributions.png")
)

print(f"\nMean confidence - Members: {np.mean(results['confidence_members']):.4f}")
print(f"Mean confidence - Non-Members: {np.mean(results['confidence_non_members']):.4f}")



plot_attack_accuracy_comparison(
    results,
    save_path=os.path.join(FIGS_DIR, f"{FIG_PREFIX}attack_metrics.png")
)

output = {
    'target_model': {
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'overfitting_gap': float(train_acc - test_acc),
    },
    'attack_results': {
        'accuracy': float(results['attack_accuracy']),
        'precision': float(results['attack_precision']),
        'recall': float(results['attack_recall']),
        'f1_score': float(results['attack_f1']),
        'auc': float(results['attack_auc']),
        'advantage': float(results['attack_accuracy'] - 0.5),
    },
    'configuration': {
        'random_seed': RANDOM_SEED,
        'num_shadow_models': SHADOW_MODEL_CONFIG['num_shadow_models'],
        'target_architecture': TARGET_MODEL_CONFIG['hidden_layers'],
        'attack_architecture': ATTACK_MODEL_CONFIG['hidden_layers'],
    }
}


results_path = os.path.join(FIGS_DIR, f"{FIG_PREFIX}attack_results.json")
with open(results_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to {results_path}")
