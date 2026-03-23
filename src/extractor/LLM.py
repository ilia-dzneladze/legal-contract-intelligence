from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

def extract_clause(contract_text: str, clause_type: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a legal document analyst. Extract the exact text from the contract that relates to the given clause type. If the clause is not present, respond with 'NOT FOUND'. Only return the extracted text, nothing else."
            },
            {
                "role": "user",
                "content": f"Clause type: {clause_type}\n\nContract:\n{contract_text}"
            }
        ],
        temperature=0,
        max_tokens=500,
    )
    return response.choices[0].message.content or ""

import json

with open("data/training/CUADv1.json") as f:
    raw = json.load(f)

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

results = []
tested = 0
max_tests = 20  # 20 contracts to stay under token budget

for article in raw["data"]:
    if tested >= max_tests:
        break
    for para in article["paragraphs"]:
        if tested >= max_tests:
            break
        if len(para["context"]) > 8000:
            continue

        for qa in para["qas"]:
            # Check if this question matches one of our clause types
            matched_clause = None
            for clause in CLAUSE_TYPES:
                if clause.lower() in qa["question"].lower():
                    matched_clause = clause
                    break
            if not matched_clause:
                continue

            gold = qa["answers"][0]["text"] if qa["answers"] else "NOT FOUND"
            predicted = extract_clause(para["context"], matched_clause)

            # Simple check: does the prediction contain the gold answer or vice versa?
            if gold == "NOT FOUND":
                correct = "NOT FOUND" in predicted.upper()
            else:
                correct = gold.strip() in predicted or predicted.strip() in gold

            results.append({
                "clause": matched_clause,
                "correct": correct,
                "gold": gold[:100],
                "predicted": predicted[:100],
            })
            
            print(f"[{len(results)}] {matched_clause}: {'✓' if correct else '✗'}")
    tested += 1

# Print results
import pandas as pd
df_results = pd.DataFrame(results)
print(f"Total comparisons: {len(df_results)}")
print(f"Accuracy: {df_results['correct'].mean():.2%}")
print()
print("Per clause type:")
print(df_results.groupby("clause")["correct"].mean().to_string())