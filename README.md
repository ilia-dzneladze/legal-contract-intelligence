# Legal Contract Intelligence

A document intelligence pipeline for classifying and extracting information from commercial legal contracts. Built on the [CUAD dataset](https://www.atticusprojectai.org/cuad) (510 contracts, 41 clause categories).

Portfolio project for an ML Engineer (NLP) Intern interview at Intapp.

---

## Pipeline

| Day | Task | Status |
|-----|------|--------|
| 1 | Contract type classification (TF-IDF baseline + Legal-BERT) | ✅ |
| 2 | Clause extraction (extractive QA) | |
| 3 | Semantic search + RAG | |
| 4 | Streamlit demo | |
| 5 | Evaluation + write-up | |

---

## Day 1: Classification

**Task:** Classify a contract into one of ~25 types (e.g. "Service Agreement", "License Agreement") from raw contract text.

**Baseline — TF-IDF + Logistic Regression**
- 5-fold cross-validation on 510 contracts
- Accuracy: 65% | Macro F1: 0.60

**Fine-tuned — `nlpaueb/legal-bert-base-uncased`**
- 80/20 train/val split, 10 epochs, fp16 on Colab T4
- Accuracy: 94% | Macro F1: 0.93

---

## Setup

```bash
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

**Run baseline:**
```bash
python -m src.classifier.baseline
```

**Run Legal-BERT fine-tuning** (requires GPU — paste into Colab):
```bash
python -m src.classifier.legal_bert_finetune
```

---

## Stack

Python · PyTorch · HuggingFace Transformers · scikit-learn · pandas
