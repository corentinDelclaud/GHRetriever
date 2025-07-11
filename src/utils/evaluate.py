import json
import pandas as pd
import re
import string


def get_accuracy_gqa(path):
    df = pd.read_json(path, lines=True)
    # compute accuracy
    correct = 0
    for pred, label in zip(df["pred"], df["label"]):
        if str(pred).strip().lower() == str(label).strip().lower():
            correct += 1
    return correct / len(df)


def get_accuracy_expla_graphs(path):
    df = pd.read_json(path, lines=True)
    # compute accuracy, precision, recall
    correct = 0
    total_pred = 0
    total_label = 0
    for pred, label in zip(df["pred"], df["label"]):
        matches = re.findall(r"support|Support|Counter|counter", pred.strip())
        if len(matches) > 0 and matches[0].lower() == label:
            correct += 1
        total_pred += 1
        total_label += 1
    accuracy = correct / len(df)
    print(f"Accuracy: {accuracy}")
    return accuracy

def get_accuracy_expla_graphs2(path):
    import numpy as np
    df = pd.read_json(path, lines=True)

    TP = 0  # True Positives
    FP = 0  # False Positives
    FN = 0  # False Negatives

    for pred, label in zip(df["pred"], df["label"]):
        # Support multi-label and multi-prediction (comma or pipe separated)
        pred_set = set([x.strip().lower() for x in re.findall(r"support|counter", pred.lower())])
        label_set = set([x.strip().lower() for x in re.findall(r"support|counter", str(label).lower())])

        TP += len(pred_set & label_set)
        FP += len(pred_set - label_set)
        FN += len(label_set - pred_set)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # For accuracy: proportion of exact matches
    exact_matches = 0
    for pred, label in zip(df["pred"], df["label"]):
        pred_set = set([x.strip().lower() for x in re.findall(r"support|counter", pred.lower())])
        label_set = set([x.strip().lower() for x in re.findall(r"support|counter", str(label).lower())])
        if pred_set == label_set:
            exact_matches += 1
    accuracy = exact_matches / len(df)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    return accuracy, precision, recall, f1


def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1


def eval_f1(prediction, answer):
    if len(prediction) == 0:
        return 0, 0, 0
    matched = 0
    prediction_str = " ".join(prediction)
    for a in answer:
        if match(prediction_str, a):
            matched += 1
    precision = matched / len(prediction)
    recall = matched / len(answer)
    if precision + recall == 0:
        return 0, precision, recall
    else:
        return 2 * precision * recall / (precision + recall), precision, recall


def eval_acc(prediction, answer):
    matched = 0.0
    for a in answer:
        if match(prediction, a):
            matched += 1
    return matched / len(answer)


def eval_hit(prediction, answer):
    for a in answer:
        if match(prediction, a):
            return 1
    return 0


def get_accuracy_webqsp(path):
    df = pd.read_json(path, lines=True)

    # Load results
    acc_list = []
    hit_list = []
    f1_list = []
    precission_list = []
    recall_list = []

    for prediction, answer in zip(df.pred.tolist(), df.label.tolist()):

        prediction = prediction.replace("|", "\n")
        answer = answer.split("|")

        prediction = prediction.split("\n")
        f1_score, precision_score, recall_score = eval_f1(prediction, answer)
        f1_list.append(f1_score)
        precission_list.append(precision_score)
        recall_list.append(recall_score)
        prediction_str = " ".join(prediction)
        acc = eval_acc(prediction_str, answer)
        hit = eval_hit(prediction_str, answer)
        acc_list.append(acc)
        hit_list.append(hit)

    acc = sum(acc_list) * 100 / len(acc_list)
    hit = sum(hit_list) * 100 / len(hit_list)
    f1 = sum(f1_list) * 100 / len(f1_list)
    pre = sum(precission_list) * 100 / len(precission_list)
    recall = sum(recall_list) * 100 / len(recall_list)

    print(f"Accuracy: {acc:.4f}")
    print(f"Hit: {hit:.4f}")
    print(f"Precision: {pre:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")

    return hit


eval_funcs = {
    "expla_graphs": get_accuracy_expla_graphs,
    "scene_graphs": get_accuracy_gqa,
    "scene_graphs_baseline": get_accuracy_gqa,
    "webqsp": get_accuracy_webqsp,
    "webqsp_baseline": get_accuracy_webqsp,
    'kg': get_accuracy_gqa,
}
