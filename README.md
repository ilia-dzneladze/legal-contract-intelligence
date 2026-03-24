# Legal Contract Intelligence

A document intelligence pipeline for classifying and extracting information from commercial legal contracts, built on the [CUAD dataset](https://www.atticusprojectai.org/cuad) (510 contracts, 41 clause categories).

---

## Contract Type Classification

### Baseline: TF-IDF + Logistic Regression
- Trained on 430 contracts across 23 types
- Model saved locally as pickle

### Fine-tuned Legal-BERT (nlpaueb/legal-bert-base-uncased)
- Fine-tuned on the same 430 contracts, 23 types
- Model available on HuggingFace: https://huggingface.co/iliadzneladze/legal-BERT-clf

### Evaluation (CUAD held-out test set — 46 contracts, 7 classes)

Evaluated on `test.json`, CUAD's official held-out set. Coverage is limited to 7 of 23 contract types due to CUAD filename conventions; per-class metrics for the remaining 16 types require additional labeled data.

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| TF-IDF + Logistic Regression | 0.9348 | 0.7236 |
| Fine-tuned Legal-BERT | **1.0000** | **1.0000** |

Legal-BERT achieves perfect scores on all 7 classes. The baseline's main failure is Development Agreement (40% recall), likely because its lexical features overlap with adjacent consulting/service contract language.

### Comparison
- **Accuracy:** TF-IDF/LogReg achieves 93.5% / 0.72 macro F1. Legal-BERT achieves 100% / 1.00 macro F1 on the held-out test set.
- **Training cost:** TF-IDF trains in seconds on CPU. Legal-BERT requires ~1 hour of GPU fine-tuning.
- **Inference:** TF-IDF runs locally in milliseconds with no dependencies beyond scikit-learn. Legal-BERT requires a GPU or the hosted HuggingFace model.
- **Flexibility:** TF-IDF struggles with overlapping contract types due to shallow lexical features. Legal-BERT handles ambiguous types well thanks to domain-specific pretraining on legal corpora.

### Usage
```python
from transformers import pipeline
clf = pipeline("text-classification", model="iliadzneladze/legal-BERT-clf")
result = clf("This agreement shall govern the franchise relationship...")
print(result)
```

---

## Information Extraction

### Fine-tuned RoBERTa (deepset/roberta-base-squad2)
- Fine-tuned on CUAD QA pairs (8 clause types, 3570 QA pairs)
- Training loss: 0.041 → 0.015 over 3 epochs
- Model available on HuggingFace: https://huggingface.co/iliadzneladze/roBERTa_cuad_finetune

### LLM-based extraction (Groq + Llama 3.1 8B)
- Zero-shot extraction, no fine-tuning required

### Evaluation (21 QA pairs from 20 contracts, 5 clause types)

Evaluated on the first 20 contracts in `CUADv1.json` with context ≤ 8000 characters. Correctness is determined by substring match (gold ⊆ predicted or predicted ⊆ gold).

| Model | Overall Accuracy |
|-------|-----------------|
| Fine-tuned RoBERTa (CUAD QA) | **90.48%** (19/21) |
| LLM — Groq / llama-3.1-8b-instant | **90.48%** (19/21) |

Both models scored identically overall, but failed on **different pairs** (RoBERTa missed #9 and #20; LLM missed #10 and #21). With only 21 samples each mistake costs ~4.8 pp, so two independent errors from either model produces the same total.

| Clause Type | RoBERTa | LLM |
|---|---|---|
| Governing Law | 100% | 100% |
| Change Of Control | 100% | 100% |
| Termination For Convenience | 100% | 100% |
| Non-Compete | 83% | 83% |
| Assignment | 83% | 83% |

### Comparison
- **Accuracy:** Both models achieve 90.5% on this evaluation subset, with errors on different examples.
- **Cost:** RoBERTa requires ~2.5 hours of GPU training but runs free at inference. LLM requires no training but costs per API call (~$0.001 per extraction).
- **Latency:** RoBERTa runs locally in milliseconds. LLM depends on API response time (~0.5–1s per call).
- **Flexibility:** LLM handles new clause types with just a prompt change. RoBERTa needs retraining for new clause types.

### Usage
```python
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_id  = "iliadzneladze/roBERTa_cuad_finetune"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model     = AutoModelForQuestionAnswering.from_pretrained(model_id)
model.eval()

question = "Highlight the parts related to Governing Law"
context  = "This agreement shall be governed by the laws of the State of Delaware."

inputs = tokenizer(question, context, return_tensors="pt", truncation="only_second", max_length=512)
with torch.no_grad():
    out = model(**inputs)

start = torch.argmax(out.start_logits).item()
end   = torch.argmax(out.end_logits).item()
print(tokenizer.decode(inputs["input_ids"][0][start:end+1], skip_special_tokens=True))
# → "the laws of the State of Delaware"
```

---

## Semantic Search + RAG

### Indexing (all-MiniLM-L6-v2 + ChromaDB)
- Contracts chunked into 500-word windows with 50-word overlap
- ~510 contracts → ~30k chunks embedded with `sentence-transformers/all-MiniLM-L6-v2`
- Embeddings stored in a persistent ChromaDB collection at `data/chroma_db/`

### RAG Pipeline (Groq + Llama 3.1 8B)
- Query → top-5 chunk retrieval via ChromaDB vector search
- Retrieved chunks passed as context to Llama 3.1 8B with a legal-analyst system prompt
- Model instructed to cite source contracts and refuse to answer out-of-context questions

### Usage
```bash
# Step 1: build the index (run once)
python -m src.search.index

# Step 2: ask questions
python -m src.search.rag
```

Example output:
```
Q: Which contracts have governing law clauses mentioning Delaware?

Retrieved chunks:
--- Chunk 1 [ContractA.pdf] ---
...the laws of the State of Delaware shall govern...

--- RAG Answer ---
Based on the excerpts, the following contracts reference Delaware as governing law:
- ContractA.pdf: "the laws of the State of Delaware shall govern..."
```

---

## Setup

```bash
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

Download the CUAD dataset from [huggingface.co/datasets/theatticusproject/cuad](https://huggingface.co/datasets/theatticusproject/cuad) and place the files in `data/training/`:
```
legal-contract-intelligence/
└── data/
    └── training/
        ├── CUADv1.json   ← training set
        └── test.json     ← test set
```

**Run baseline** (CPU, no GPU needed):
```bash
python -m src.classifier.baseline
```

**Run Legal-BERT fine-tuning** (GPU required):
```bash
python -m src.classifier.legal_bert_finetune
```

**Run RoBERTa QA fine-tuning** (GPU required):
```bash
python -m src.extractor.roberta_cuad_finetune
```

**Run LLM-based extraction:**

Requires a Groq API key — get one free at [console.groq.com](https://console.groq.com). Add it to a `.env` file in the repo root:
```
GROQ_API_KEY=your_key_here
```
```bash
python -m src.extractor.LLM
```

**Run evaluations:**
```bash
python -m src.classifier.evaluate
python -m src.extractor.evaluate
```

> `fp16` mixed-precision is enabled automatically when a CUDA GPU is detected, and disabled on CPU.

> **No GPU?** Legal-BERT and RoBERTa fine-tuning each take ~1–2.5 hours and require a CUDA GPU. Use [Google Colab](https://colab.research.google.com) (free T4) or [Kaggle Notebooks](https://www.kaggle.com/code) (free T4/P100) to run them. The TF-IDF baseline, LLM extraction, and all evaluation scripts run fine on CPU.

---

## Stack

| Tool | Purpose |
|------|---------|
| PyTorch | Model training and inference |
| HuggingFace Transformers | Legal-BERT and RoBERTa fine-tuning |
| scikit-learn | TF-IDF baseline |
| sentence-transformers | Chunk embeddings for semantic search |
| ChromaDB | Vector store for RAG retrieval |
| Groq API + Llama 3.1 8B | LLM-based extraction and RAG generation |
| pandas | Data loading and preprocessing |
