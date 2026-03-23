# Legal Contract Intelligence

A document intelligence pipeline for classifying and extracting information from commercial legal contracts. Built on the [CUAD dataset](https://www.atticusprojectai.org/cuad) (510 contracts, 41 clause categories).

Portfolio project for an ML Engineer (NLP) Intern interview at Intapp.

---

## Pipeline

| Day | Task | Status |
|-----|------|--------|
| 1 | Contract type classification (TF-IDF baseline + Legal-BERT) | ✅ |
| 2 | Clause extraction (extractive QA) | ✅ |
| 3 | Semantic search + RAG | |
| 4 | Streamlit demo | |
| 5 | Evaluation + write-up | |

---

## Day 1: Contract Type Classification

### Baseline: TF-IDF + Logistic Regression
- Cross-validation accuracy: 65%, macro F1: 0.60
- Trained on ~430 contracts across 23 types
- Model saved locally as pickle

### Fine-tuned Legal-BERT (nlpaueb/legal-bert-base-uncased)
- Validation accuracy: 94%, macro F1: 0.93
- Massive improvement over baseline, especially on overlapping types
- Model available on HuggingFace: https://huggingface.co/iliadzneladze/legal-BERT-clf

To test the classifier:
```python
from transformers import pipeline
clf = pipeline("text-classification", model="iliadzneladze/legal-BERT-clf")
result = clf("This agreement shall govern the franchise relationship...")
print(result)
```


## Day 2: Information Extraction

### Approach A: Fine-tuned RoBERTa (deepset/roberta-base-squad2)
- Fine-tuned on CUAD QA pairs (8 clause types, 3570 QA pairs)
- Training loss: 0.041 → 0.015 over 3 epochs
- Model lost due to Colab runtime disconnect before evaluation
- To be retrained and evaluated when compute is available

### Approach B: LLM-based extraction (Groq + Llama 3.1 8B)
- Zero-shot extraction, no fine-tuning required
- Overall accuracy: 90.5% (19/21 correct)
- Tested on 5 of 8 clause types across 20 contracts
- Per-clause accuracy:
  - Governing Law: 100%
  - Change of Control: 100%
  - Termination for Convenience: 100%
  - Non-Compete: 83%
  - Assignment: 83%

### Comparison
- **Accuracy:** RoBERTa pending evaluation. LLM achieves 90.5% with no training.
- **Cost:** RoBERTa requires ~2.5 hours of GPU training but runs free at inference. LLM requires no training but costs per API call (~0.001 per extraction).
- **Latency:** RoBERTa runs locally in milliseconds. LLM depends on API response time (~0.5-1s per call).
- **Flexibility:** LLM handles new clause types with just a prompt change. RoBERTa needs retraining for new clause types.

---

## Setup

### Local

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
        └── test.json     ← test set (required by baseline)
```

**Run baseline** (CPU fine, no GPU needed):
```bash
python -m src.classifier.baseline
```
The trained pipeline is saved to `models/baseline_tfidf_logreg.pkl`.

**Run Legal-BERT fine-tuning** (GPU recommended):
```bash
python -m src.classifier.legal_bert_finetune
```
Checkpoints and the final model are saved to `model/legal-bert-clf/`.

**Run RoBERTa QA fine-tuning** (GPU recommended):
```bash
python -m src.extractor.roberta_cuad_finetune
```
Checkpoints and the final model are saved to `model/roberta-cuad-qa/`.

> `fp16` mixed-precision is enabled automatically when a CUDA GPU is detected, and disabled on CPU so both scripts run locally without modification.

---

### Google Colab

1. Mount your Drive and clone the repo:
```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive
!git clone <your-repo-url> legal-contract-intelligence
%cd legal-contract-intelligence
```

2. Install dependencies:
```bash
!pip install -r requirements.txt
```

3. Upload `CUADv1.json` and `test.json` to `data/training/` in the repo folder on your Drive.

4. Run any of the training scripts:
```bash
# TF-IDF baseline (no GPU needed)
!python -m src.classifier.baseline

# Contract type classifier
!python -m src.classifier.legal_bert_finetune

# Clause extraction QA model
!python -m src.extractor.roberta_cuad_finetune
```

Checkpoints are saved directly to your Drive (`MyDrive/legal-bert-clf/` and `MyDrive/roberta-cuad-qa/`) and persist after the runtime disconnects.

> To point either script at a different copy of `CUADv1.json`, set the `CUAD_PATH` environment variable:
> ```bash
> CUAD_PATH=/path/to/CUADv1.json python -m src.classifier.legal_bert_finetune
> ```

---

### LLM-based Clause Extraction (Groq)

No GPU required. The script calls the Groq API and evaluates against CUAD ground truth.

**1. Get a free Groq API key** at [console.groq.com](https://console.groq.com).

**2. Create a `.env` file in the repo root:**
```
GROQ_API_KEY=your_key_here
```

**3. Make sure `CUADv1.json` is in `data/training/` (same as above), then run:**
```bash
python -m src.extractor.LLM
```

Expected output — per-extraction results followed by a summary:
```
[1] Governing Law: ✓
[2] Change Of Control: ✓
...
Total comparisons: 21
Accuracy: 90.48%

Per clause type:
Assignment                      0.833333
Change Of Control               1.000000
Governing Law                   1.000000
...
```

**Colab:** set the key as a Colab secret (Secrets panel → `GROQ_API_KEY`) and load it before running:
```python
import os
from google.colab import userdata
os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")

!python -m src.extractor.LLM
```

---

## Stack

Python | PyTorch | HuggingFace Transformers | scikit-learn | pandas
