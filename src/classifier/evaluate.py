import os
import sys
import pickle
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, "../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.data_loader import load_data
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Data
# test.json is CUAD's official held-out set — the only data neither model has
# seen during training. It covers 7/23 contract types due to CUAD filename
# conventions, so per-class metrics for the remaining 16 types are not available.

print("Loading test data...")
test_df = load_data("test.json")
test_texts = test_df["text"].tolist()
test_labels = test_df["contract_type"].tolist()
print(f"Eval set: {len(test_df)} contracts, {len(set(test_labels))} classes\n")

# Model 1: TF-IDF + Logistic Regression 

print("Model 1: TF-IDF + Logistic Regression (Baseline)\n")

baseline_path = os.path.join(_repo_root, "models", "baseline_tfidf_logreg.pkl")
with open(baseline_path, "rb") as f:
    baseline_model = pickle.load(f)

baseline_preds = baseline_model.predict(test_texts)
baseline_acc = accuracy_score(test_labels, baseline_preds)
baseline_f1 = f1_score(test_labels, baseline_preds, average="macro", zero_division=0)

print(f"Accuracy : {baseline_acc:.4f}")
print(f"Macro F1 : {baseline_f1:.4f}")
print("\nClassification Report:")
print(classification_report(test_labels, baseline_preds, zero_division=0))

# Model 2: Fine-tuned Legal-BERT 

print("Model 2: Fine-tuned Legal-BERT\n")

bert_path = os.path.join(_repo_root, "models", "legal-bert-clf-final")
tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForSequenceClassification.from_pretrained(bert_path)
bert_model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
bert_model.to(device)
print(f"Running inference on: {device}")

BATCH_SIZE = 16
bert_preds = []

for i in range(0, len(test_texts), BATCH_SIZE):
    batch = test_texts[i : i + BATCH_SIZE]
    encoded = tokenizer(
        batch,
        truncation=True,
        max_length=512,
        padding=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        logits = bert_model(**encoded).logits
    ids = torch.argmax(logits, dim=-1).cpu().tolist()
    bert_preds.extend([bert_model.config.id2label[p] for p in ids])

bert_acc = accuracy_score(test_labels, bert_preds)
bert_f1 = f1_score(test_labels, bert_preds, average="macro", zero_division=0)

print(f"Accuracy : {bert_acc:.4f}")
print(f"Macro F1 : {bert_f1:.4f}")
print("\nClassification Report:")
print(classification_report(test_labels, bert_preds, zero_division=0))

# Summary

print("COMPARISON SUMMARY\n")
print(f"{'Model':<40} {'Accuracy':>10} {'Macro F1':>10}\n")
print(f"{'TF-IDF + Logistic Regression':<40} {baseline_acc:>10.4f} {baseline_f1:>10.4f}")
print(f"{'Fine-tuned Legal-BERT':<40} {bert_acc:>10.4f} {bert_f1:>10.4f}")
