import os
import sys
import torch
import numpy as np

# Allow running as a script from any CWD (Colab, repo root, etc.)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, "../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.data_loader import load_data
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.metrics import classification_report, f1_score, accuracy_score

# get data ready

# Place CUADv1.json in data/training/ relative to repo root, or set CUAD_PATH env var.
contracts = load_data(os.environ.get("CUAD_PATH", "CUADv1.json"))

labels = sorted(contracts["contract_type"].unique())
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

contracts["label"] = contracts["contract_type"].map(label2id)

train_df, val_df = train_test_split(
    contracts, test_size=0.2, random_state=42,
    stratify=contracts["label"]
)
print(f"Train: {len(train_df)}, Val: {len(val_df)}")

# tokenize

tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

train_dataset = Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_df[["text", "label"]].reset_index(drop=True))

train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
val_dataset = val_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

train_dataset.set_format("torch")
val_dataset.set_format("torch")

# train model

model = AutoModelForSequenceClassification.from_pretrained(
    "nlpaueb/legal-bert-base-uncased",
    num_labels=len(labels),
    label2id=label2id,
    id2label=id2label,
)

def compute_metrics(eval_pred):
    logits, label_ids = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(label_ids, preds),
        "macro_f1": f1_score(label_ids, preds, average="macro"),
    }

_on_colab = os.path.isdir("/content")
_output_dir = (
    "/content/drive/MyDrive/legal-bert-clf" if _on_colab
    else os.path.join(_repo_root, "model/legal-bert-clf")
)
_use_fp16 = torch.cuda.is_available()

training_args = TrainingArguments(
    output_dir=_output_dir,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    fp16=_use_fp16,
    report_to="none",
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# output metrics

preds_output = trainer.predict(val_dataset) # type: ignore
preds = np.argmax(preds_output.predictions, axis=-1)
true_labels = val_df["label"].values

print(classification_report(true_labels, preds, target_names=labels))

_final_dir = os.path.join(_output_dir, "final")
trainer.save_model(_final_dir)
tokenizer.save_pretrained(_final_dir)