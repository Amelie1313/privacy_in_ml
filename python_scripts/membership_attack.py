import os
import pickle

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

from cli_utils import print_banner
from dataset_config import DEFAULT_RESULTS_DIR, get_config
from perturbations import load_dataset


def evaluate_true_class_confidence(confidence_scores, y_train, y_test, class_order):
    """For each model (one per epsilon key in confidence_scores), looks up
    the confidence it assigned to each sample's *actual* label and reports
    confidence/loss/entropy gaps between training-set members and held-out
    non-members - the core membership-inference signal."""
    true_labels = np.concatenate([y_train, y_test])
    class_order = np.array(class_order)

    membership = np.concatenate([np.ones(len(y_train)), np.zeros(len(y_test))])

    all_mean, member_mean, nonmember_mean = [], [], []
    member_loss_mean, nonmember_loss_mean = [], []
    member_entropy_mean, nonmember_entropy_mean = [], []

    for key, scores in confidence_scores.items():
        true_class_indices = np.array([np.where(class_order == label)[0][0] for label in true_labels])
        true_confidences = scores[np.arange(len(true_labels)), true_class_indices]

        member_confidence = true_confidences[:len(y_train)]
        nonmember_confidence = true_confidences[len(y_train):]

        loss = -np.log(np.clip(true_confidences, 1e-10, 1.0))
        member_loss = loss[:len(y_train)]
        nonmember_loss = loss[len(y_train):]

        auc = roc_auc_score(membership, true_confidences)
        auc_loss = roc_auc_score(membership, -loss)

        entropy = -np.sum(scores * np.log(np.clip(scores, 1e-10, 1.0)), axis=1)
        member_entropy = entropy[:len(y_train)]
        nonmember_entropy = entropy[len(y_train):]
        auc_entropy = roc_auc_score(membership, -entropy)

        print('-----------------')
        print(f"Model {key}")
        print('-----------------')
        print("Confidence:")
        print("  Overall mean:", true_confidences.mean())
        print("  Member mean:", member_confidence.mean())
        print("  Non-member mean:", nonmember_confidence.mean())
        print("  Confidence gap:", member_confidence.mean() - nonmember_confidence.mean())
        print("  MIA AUC:", auc)
        print()
        print("Loss:")
        print("  Overall mean:", loss.mean())
        print("  Member mean:", member_loss.mean())
        print("  Non-member mean:", nonmember_loss.mean())
        print("  Loss gap:", nonmember_loss.mean() - member_loss.mean())
        print("  MIA AUC:", auc_loss)
        print()
        print("Entropy:")
        print("  Overall mean:", entropy.mean())
        print("  Member mean:", member_entropy.mean())
        print("  Non-member mean:", nonmember_entropy.mean())
        print("  Entropy gap:", nonmember_entropy.mean() - member_entropy.mean())
        print("  MIA AUC:", auc_entropy)
        print()
        print("Confidence:")
        print("  Minimum:", true_confidences.min())
        print("  Maximum:", true_confidences.max())
        print('=========================================\n')

        all_mean.append(true_confidences.mean())
        member_mean.append(member_confidence.mean())
        nonmember_mean.append(nonmember_confidence.mean())
        member_loss_mean.append(member_loss.mean())
        nonmember_loss_mean.append(nonmember_loss.mean())
        member_entropy_mean.append(member_entropy.mean())
        nonmember_entropy_mean.append(nonmember_entropy.mean())

    print("=== Across all models ===")
    print("Average overall confidence:", np.mean(all_mean))
    print("Average member confidence:", np.mean(member_mean))
    print("Average non-member confidence:", np.mean(nonmember_mean))
    print()
    print("Average member loss:", np.mean(member_loss_mean))
    print("Average non-member loss:", np.mean(nonmember_loss_mean))
    print()
    print("Average member entropy:", np.mean(member_entropy_mean))
    print("Average non-member entropy:", np.mean(nonmember_entropy_mean))


def evaluate_true_max_confidence(confidence_scores, y_train, y_test):
    """Same as evaluate_true_class_confidence, but uses each sample's max
    predicted-class confidence rather than the confidence of its true class
    (doesn't need class_order - the attacker doesn't need to know the label)."""
    membership = np.concatenate([np.ones(len(y_train)), np.zeros(len(y_test))])

    all_mean, member_mean, nonmember_mean = [], [], []
    member_loss_mean, nonmember_loss_mean = [], []
    member_entropy_mean, nonmember_entropy_mean = [], []

    for key, scores in confidence_scores.items():
        true_confidences = scores.max(axis=1)
        member_confidence = true_confidences[:len(y_train)]
        nonmember_confidence = true_confidences[len(y_train):]

        loss = -np.log(np.clip(true_confidences, 1e-10, 1.0))
        member_loss = loss[:len(y_train)]
        nonmember_loss = loss[len(y_train):]

        auc = roc_auc_score(membership, true_confidences)
        auc_loss = roc_auc_score(membership, -loss)

        entropy = -np.sum(scores * np.log(np.clip(scores, 1e-10, 1.0)), axis=1)
        member_entropy = entropy[:len(y_train)]
        nonmember_entropy = entropy[len(y_train):]
        auc_entropy = roc_auc_score(membership, -entropy)

        print('-----------------')
        print(f"Model {key}")
        print('-----------------')
        print("Confidence:")
        print("  Overall mean:", true_confidences.mean())
        print("  Member mean:", member_confidence.mean())
        print("  Non-member mean:", nonmember_confidence.mean())
        print("  Confidence gap:", member_confidence.mean() - nonmember_confidence.mean())
        print("  MIA AUC:", auc)
        print()
        print("Loss:")
        print("  Overall mean:", loss.mean())
        print("  Member mean:", member_loss.mean())
        print("  Non-member mean:", nonmember_loss.mean())
        print("  Loss gap:", nonmember_loss.mean() - member_loss.mean())
        print("  MIA AUC:", auc_loss)
        print()
        print("Entropy:")
        print("  Overall mean:", entropy.mean())
        print("  Member mean:", member_entropy.mean())
        print("  Non-member mean:", nonmember_entropy.mean())
        print("  Entropy gap:", nonmember_entropy.mean() - member_entropy.mean())
        print("  MIA AUC:", auc_entropy)
        print()
        print("Confidence:")
        print("  Minimum:", true_confidences.min())
        print("  Maximum:", true_confidences.max())
        print('=========================================\n')

        all_mean.append(true_confidences.mean())
        member_mean.append(member_confidence.mean())
        nonmember_mean.append(nonmember_confidence.mean())
        member_loss_mean.append(member_loss.mean())
        nonmember_loss_mean.append(nonmember_loss.mean())
        member_entropy_mean.append(member_entropy.mean())
        nonmember_entropy_mean.append(nonmember_entropy.mean())

    print("=== Across all models ===")
    print("Average overall confidence:", np.mean(all_mean))
    print("Average member confidence:", np.mean(member_mean))
    print("Average non-member confidence:", np.mean(nonmember_mean))
    print()
    print("Average member loss:", np.mean(member_loss_mean))
    print("Average non-member loss:", np.mean(nonmember_loss_mean))
    print()
    print("Average member entropy:", np.mean(member_entropy_mean))
    print("Average non-member entropy:", np.mean(nonmember_entropy_mean))


def evaluate_threshold_attack(confidence_scores, y_train, y_test, threshold):
    """The actual attack: predicts "this sample was a training member" iff
    the model's max confidence on it exceeds `threshold`."""
    membership = np.concatenate([np.ones(len(y_train)), np.zeros(len(y_test))])
    for key, scores in confidence_scores.items():
        confidence = scores.max(axis=1)
        attack_labels = np.array([1 if conf > threshold else 0 for conf in confidence])

        print('-----------------')
        print(f"Model {key}")
        print('-----------------')
        print("Threshold:", threshold)
        print("Predicted members:", np.sum(attack_labels == 1))
        print("Predicted non-members:", np.sum(attack_labels == 0))
        print("Actual members:", np.sum(membership == 1))
        print("Actual non-members:", np.sum(membership == 0))
        print("Accuracy:", accuracy_score(membership, attack_labels))
        print("Confusion matrix:")
        print(confusion_matrix(membership, attack_labels))
        print('=========================================\n')


def _load_pickle(pkl_dir, filename):
    path = os.path.join(pkl_dir, filename)
    if not os.path.exists(path):
        print(f"  (skipping - {path} not found; run the matching perturbation section first)")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def run_membership_attack(
    dataset_name,
    sections=("input", "output", "internal"),
    variant="default",
    threshold=None,
    data_dir=None,
    results_dir=DEFAULT_RESULTS_DIR,
    pkl_dir=None,
):
    """Loads the confidence-score pickles produced by perturbations.py (or
    row_removal.py) for `dataset_name` and runs all three attack evaluations
    against each requested section."""
    from dataset_config import DEFAULT_DATA_DIR, DEFAULT_ATTACK_THRESHOLD

    cfg = get_config(dataset_name)
    data_dir = data_dir or DEFAULT_DATA_DIR
    pkl_dir = pkl_dir or str(results_dir)
    threshold = DEFAULT_ATTACK_THRESHOLD if threshold is None else threshold

    _, _, y_train, y_test = load_dataset(dataset_name, variant=variant, data_dir=data_dir)

    pkl_prefixes = {"input": "input", "output": "output", "internal": "internal", "original": "orig"}

    section_titles = {
        "original": "ORIGINAL (BASELINE) MODEL",
        "input": "INPUT-PERTURBED MODELS",
        "output": "OUTPUT-PERTURBED MODELS",
        "internal": "INTERNAL (OBJECTIVE)-PERTURBED MODELS",
    }

    for section in ["original", "input", "output", "internal"]:
        if section not in sections:
            continue
        print_banner(f"MEMBERSHIP ATTACK - {section_titles[section]} - {dataset_name}")
        filename = f"{pkl_prefixes[section]}_confidence_scores_{cfg['pkl_name']}.pkl"
        confidence_scores = _load_pickle(pkl_dir, filename)
        if confidence_scores is None:
            continue

        # orig_confidence_scores.pkl is a single array (no epsilon axis) -
        # wrap it so it fits the same {key: scores} shape as the others.
        if not isinstance(confidence_scores, dict):
            confidence_scores = {"Original": confidence_scores}

        print("--- Evaluation 1/3: confidence in the sample's TRUE class ---\n")
        evaluate_true_class_confidence(confidence_scores, y_train, y_test, cfg["class_order"])

        print("\n--- Evaluation 2/3: confidence in the model's MAX-predicted class ---\n")
        evaluate_true_max_confidence(confidence_scores, y_train, y_test)

        print(f"\n--- Evaluation 3/3: threshold attack (threshold={threshold}) ---\n")
        evaluate_threshold_attack(confidence_scores, y_train, y_test, threshold=threshold)
