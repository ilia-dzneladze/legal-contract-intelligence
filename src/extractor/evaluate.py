import os
import sys
import json
import torch
import pandas as pd
from dotenv import load_dotenv

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root   = os.path.abspath(os.path.join(_script_dir, "../.."))

load_dotenv(os.path.join(_repo_root, ".env"))

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

# Build evaluation set (same methodology as LLM.py) 

CUAD_PATH    = os.path.join(_repo_root, "data", "training", "CUADv1.json")
MAX_CONTRACTS = 20

print("Loading CUAD data...")
with open(CUAD_PATH) as f:
    raw = json.load(f)

eval_pairs = []
tested = 0

for article in raw["data"]:
    if tested >= MAX_CONTRACTS:
        break
    for para in article["paragraphs"]:
        if tested >= MAX_CONTRACTS:
            break
        if len(para["context"]) > 8000:
            continue
        for qa in para["qas"]:
            matched_clause = next(
                (c for c in CLAUSE_TYPES if c.lower() in qa["question"].lower()),
                None,
            )
            if not matched_clause:
                continue
            gold = qa["answers"][0]["text"] if qa["answers"] else "NOT FOUND"
            eval_pairs.append({
                "question": qa["question"],
                "context":  para["context"],
                "clause":   matched_clause,
                "gold":     gold,
            })
    tested += 1

print(f"Evaluation set: {len(eval_pairs)} QA pairs from {MAX_CONTRACTS} contracts\n")


def is_correct(gold: str, predicted: str) -> bool:
    if gold == "NOT FOUND":
        return "NOT FOUND" in predicted.upper()
    return gold.strip() in predicted or predicted.strip() in gold


# Model 1: Fine-tuned RoBERTa CUAD QA 

print("Model 1: Fine-tuned RoBERTa CUAD QA\n")

from transformers import AutoTokenizer, AutoModelForQuestionAnswering

roberta_path = os.path.join(_repo_root, "models", "roberta-cuad-qa-final")
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
qa_tokenizer = AutoTokenizer.from_pretrained(roberta_path)
qa_model     = AutoModelForQuestionAnswering.from_pretrained(roberta_path).to(device)
qa_model.eval()


def predict_span(question: str, context: str) -> str:
    inputs = qa_tokenizer(
        question, context,
        truncation="only_second",
        max_length=512,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        out = qa_model(**inputs)
    start = torch.argmax(out.start_logits).item()
    end   = torch.argmax(out.end_logits).item()
    # end before start or both pointing to CLS (token 0) → no answer
    if end < start or (start == 0 and end == 0):
        return "NOT FOUND"
    tokens = inputs["input_ids"][0][start : end + 1]
    return qa_tokenizer.decode(tokens, skip_special_tokens=True).strip() or "NOT FOUND"


roberta_results = []
for i, pair in enumerate(eval_pairs):
    predicted = predict_span(pair["question"], pair["context"])
    correct   = is_correct(pair["gold"], predicted)
    roberta_results.append({"clause": pair["clause"], "correct": correct})
    print(f"[{i+1:2}/{len(eval_pairs)}] {pair['clause']}: {'✓' if correct else '✗'}")

df_roberta   = pd.DataFrame(roberta_results)
roberta_acc  = df_roberta["correct"].mean()

print(f"\nOverall Accuracy: {roberta_acc:.2%}")
print("Per clause type:")
print(df_roberta.groupby("clause")["correct"].mean().to_string())

# Model 2: LLM (Groq / llama-3.1-8b-instant) 

print("\nModel 2: LLM Extraction (Groq / llama-3.1-8b-instant)\n")

groq_key = os.environ.get("GROQ_API_KEY")
llm_acc  = None

if not groq_key:
    print("GROQ_API_KEY not set — skipping LLM evaluation.")
else:
    from groq import Groq

    client = Groq(api_key=groq_key)

    def extract_clause(contract_text: str, clause_type: str) -> str:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a legal document analyst. Extract the exact text "
                        "from the contract that relates to the given clause type. "
                        "If the clause is not present, respond with 'NOT FOUND'. "
                        "Only return the extracted text, nothing else."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Clause type: {clause_type}\n\nContract:\n{contract_text}",
                },
            ],
            temperature=0,
            max_tokens=500,
        )
        return response.choices[0].message.content or ""

    llm_results = []
    for i, pair in enumerate(eval_pairs):
        predicted = extract_clause(pair["context"], pair["clause"])
        correct   = is_correct(pair["gold"], predicted)
        llm_results.append({"clause": pair["clause"], "correct": correct})
        print(f"[{i+1:2}/{len(eval_pairs)}] {pair['clause']}: {'✓' if correct else '✗'}")

    df_llm  = pd.DataFrame(llm_results)
    llm_acc = df_llm["correct"].mean()

    print(f"\nOverall Accuracy: {llm_acc:.2%}")
    print("Per clause type:")
    print(df_llm.groupby("clause")["correct"].mean().to_string())

# Summary 

print("\nCOMPARISON SUMMARY\n")
print(f"{'Model':<45} {'Accuracy':>10}")
print("\n")
print(f"{'Fine-tuned RoBERTa (CUAD QA)':<45} {roberta_acc:>10.2%}")
if llm_acc is not None:
    print(f"{'LLM (Groq / llama-3.1-8b-instant)':<45} {llm_acc:>10.2%}")
else:
    print(f"{'LLM (Groq / llama-3.1-8b-instant)':<45} {'N/A (no API key)':>10}")
