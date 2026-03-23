import os
import pandas as pd
import json

_data_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../data/training")
)

KNOWN_TYPES = [
    "Affiliate Agreement",
    "Agency Agreement",
    "Co-Branding Agreement",
    "Collaboration Agreement",
    "Consulting Agreement",
    "Development Agreement",
    "Distributor Agreement",
    "Endorsement Agreement",
    "Franchise Agreement",
    "Hosting Agreement",
    "IP Agreement",
    "Joint Venture Agreement",
    "License Agreement",
    "Maintenance Agreement",
    "Manufacturing Agreement",
    "Marketing Agreement",
    "Non-Compete Agreement",
    "Outsourcing Agreement",
    "Promotion Agreement",
    "Reseller Agreement",
    "Service Agreement",
    "Sponsorship Agreement",
    "Strategic Alliance Agreement",
    "Supply Agreement",
    "Transportation Agreement",
]

def load_data(filename):
    with open(os.path.join(_data_dir, filename)) as f:
        raw = json.load(f)

    rows = []
    for article in raw["data"]:
        title = article["title"]
        for para in article["paragraphs"]:
            context = para["context"]
            rows.append({"title": title, "context": context})

    df = pd.DataFrame(rows)

    contracts = df.groupby("title")["context"].apply(lambda x: "\n".join(x)).reset_index()
    contracts.columns = ["filename", "text"]

    contracts["contract_type"] = "Unknown"

    for i, row in contracts.iterrows():
        for cur_type in KNOWN_TYPES:
            if cur_type.lower() in row["filename"].lower():
                contracts.at[i, "contract_type"] = cur_type
                break

    type_counts = contracts["contract_type"].value_counts()
    valid_types = type_counts[type_counts >= 5].index
    contracts = contracts[
        (contracts["contract_type"].isin(valid_types)) & 
        (contracts["contract_type"] != "Unknown")
    ]

    return contracts.reset_index(drop=True)

if __name__ == "__main__":
    print(load_data("CUADv1.json"))
        