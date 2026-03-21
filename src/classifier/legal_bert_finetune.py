from src.classifier.data_loader import load_data
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import classification_report, f1_score, accuracy_score

# get data ready

contracts = load_data("CUADv1.json")

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

training_args = TrainingArguments(
    output_dir="./legal-bert-clf",
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
    fp16=True,
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

trainer.save_model("./legal-bert-clf-final")
tokenizer.save_pretrained("./legal-bert-clf-final")