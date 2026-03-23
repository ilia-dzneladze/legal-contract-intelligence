import pandas as pd
from src.data_loader import load_data

contracts = load_data("CUADv1.json")

# Chunking
def chunk_contracts(contracts: pd.DataFrame, chunk_size: int=500, overlap: int=50):
    chunks = []
    for _, row in contracts.iterrows():
        words = row['text'].split()
        for start in range(0, len(words), chunk_size - overlap):
            chunk_text = " ".join(words[start:start + chunk_size])
            if len(chunk_text.strip()) < 50:
                continue
            chunks.append({
                "text": chunk_text,
                "filename": row["filename"],
                "contract_type": row["contract_type"],
                "chunk_id": f"{row['filename']}_chunk_{start}"
            })
    return chunks

chunks = chunk_contracts(contracts)
print(f"Contracts: {len(contracts)}, Chunks: {len(chunks)}")

# Embedding the chunks
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [chunk["text"] for chunk in chunks]
embeddings = model.encode(texts, show_progress_bar=True)

# Store the embeddings + chunks in ChromaDB
client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_or_create_collection("contracts")

batch_size = 5000
for start in range(0, len(chunks), batch_size):
    end = min(start + batch_size, len(chunks))
    collection.add(
        ids=[chunk["chunk_id"] for chunk in chunks[start:end]],
        documents=texts[start:end],
        embeddings=embeddings[start:end].tolist(),
        metadatas=[{"filename": chunk["filename"], "contract_type": chunk["contract_type"]} for chunk in chunks[start:end]],
    )
    print(f"Added batch {start}-{end}")

print(f"Stored {len(chunks)} chunks in ChromaDB")

results = collection.query(
    query_texts=["What is the governing law?"],
    n_results=3,
)

for i, doc in enumerate(results["documents"][0]): # type: ignore
    print(f"\n--- Result {i+1} ---")
    print(f"Contract: {results['metadatas'][0][i]['filename']}") # type: ignore
    print(f"Text: {doc[:200]}")