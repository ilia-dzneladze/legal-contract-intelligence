import chromadb
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])
chroma = chromadb.PersistentClient(path="data/chroma_db")
collection = chroma.get_collection("contracts")

def search(query: str, n_results: int = 5):
    results = collection.query(query_texts=[query], n_results=n_results)
    return results

def rag_answer(question: str) -> str:
    results = search(question)
    
    # Build context from retrieved chunks
    context = ""
    for i, doc in enumerate(results["documents"][0]): # type: ignore
        source = results["metadatas"][0][i]["filename"] # type: ignore
        context += f"\n[Source: {source}]\n{doc[:1000]}\n"

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a legal contract analyst. Answer the question based ONLY on the provided contract excerpts. Cite which contract each piece of information comes from. If the answer is not in the excerpts, say so."
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nContract excerpts:\n{context}"
            }
        ],
        temperature=0,
        max_tokens=500,
    )
    return response.choices[0].message.content # type: ignore

if __name__ == "__main__":
    question = "Which contracts have governing law clauses mentioning Delaware?"
    print(f"Q: {question}\n")
    
    results = search(question)
    print("Retrieved chunks:")
    for i, doc in enumerate(results["documents"][0]): # type: ignore
        source = results["metadatas"][0][i]["filename"] # type: ignore
        print(f"\n--- Chunk {i+1} [{source}] ---")
        print(doc[:200])
    
    print("\n\n--- RAG Answer ---")
    print(rag_answer(question))