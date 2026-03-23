import json
import os
import torch

# Whatever is interesting to us
CLAUSE_TYPES = [
    "Governing Law",
    "Termination For Convenience",
    "Non-Compete",
    "Change Of Control",
    "Assignment",
    "Indemnification",
    "Limitation Of Liability",
    "Intellectual Property Ownership",
]

# Place CUADv1.json in the same directory as this script, or set CUAD_PATH env var.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_cuad_path = os.environ.get("CUAD_PATH", os.path.join(_script_dir, "CUADv1.json"))

with open(_cuad_path) as f:
  raw = json.load(f)

qa_data = []
for article in raw["data"]:
  for para in article["paragraphs"]:
    context = para["context"]
    for qa in para["qas"]:
      question = qa["question"]

      if not any(clause.lower() in question.lower() for clause in CLAUSE_TYPES):
        continue

      is_impossible = qa.get("is_impossible", len(qa["answers"]) == 0)
      if is_impossible:
        qa_data.append({
          "context": context,
          "question": question,
          "answers": {"text": [], "answer_start": []},
        })
      else:
        qa_data.append({
          "context": context,
          "question": question,
          "answers": {
            "text": [a["text"] for a in qa["answers"]],
            "answer_start": [a["answer_start"] for a in qa["answers"]],
          },
        })

from sklearn.model_selection import train_test_split
from datasets import Dataset

train_data, val_data = train_test_split(qa_data, test_size=0.2, random_state=42)
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")

def preprocess(examples):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = tokenized.pop("offset_mapping")
    sample_map = tokenized.pop("overflow_to_sample_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answers = examples["answers"][sample_idx]

        # No answer - point to position 0 (the CLS token)
        if len(answers["answer_start"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        # Find where the context tokens start and end
        sequence_ids = tokenized.sequence_ids(i)
        context_start = next(j for j, s in enumerate(sequence_ids) if s == 1)
        context_end = len(sequence_ids) - 1 - next(
            j for j, s in enumerate(reversed(sequence_ids)) if s == 1
        )

        # Answer is outside this chunk - label as no answer
        if offsets[context_start][0] > start_char or offsets[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Find the token that contains the answer start character
            token_start = next(
                (j for j in range(context_start, context_end + 1)
                 if offsets[j][0] <= start_char < offsets[j][1]),
                context_start,
            )
            # Find the token that contains the answer end character
            token_end = next(
                (j for j in range(context_start, context_end + 1)
                 if offsets[j][0] < end_char <= offsets[j][1]),
                context_end,
            )
            start_positions.append(token_start)
            end_positions.append(token_end)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized

train_tokenized = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names)
val_tokenized = val_dataset.map(preprocess, batched=True, remove_columns=val_dataset.column_names)

from transformers import AutoModelForQuestionAnswering
model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

from transformers import TrainingArguments, Trainer

_on_colab = os.path.isdir("/content")
_output_dir = (
    "/content/drive/MyDrive/roberta-cuad-qa" if _on_colab
    else os.path.join(_script_dir, "../../model/roberta-cuad-qa")
)
_use_fp16 = torch.cuda.is_available()
# RoBERTa + 512 tokens: safe batch size is 8–16 on a 16 GB GPU.
# 32 will OOM on a T4. Increase only if you have an A100/H100.
_batch_size = 16

training_args = TrainingArguments(
    output_dir=_output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=_batch_size,
    per_device_eval_batch_size=_batch_size,
    learning_rate=3e-5,
    weight_decay=0.01,
    warmup_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=_use_fp16,
    report_to="none",
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
)

trainer.train()

_final_dir = os.path.join(_output_dir, "final")
trainer.save_model(_final_dir)
tokenizer.save_pretrained(_final_dir)