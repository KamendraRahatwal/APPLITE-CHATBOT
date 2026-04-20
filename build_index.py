import pandas as pd
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading Excel...")
df = pd.read_excel("faq.xlsx")

records = []
texts = []

for _, row in df.iterrows():
    record = {
        "intent": row["INTENT"] if "INTENT" in row else row.get("intent", ""),
        "keywords": row["KEYWORDS"] if "KEYWORDS" in row else row.get("keywords", ""),
        "question": row["QUESTION"] if "QUESTION" in row else row.get("question", ""),
        "answer": row["ANSWER"] if "ANSWER" in row else row.get("answer", ""),
        "follow_up_questions": row.get("FOLLOW UP QUESTIONS", "")  # NEW: Read follow-up questions
    }
    
    records.append(record)
    
    # Create embedding text
    combined = f"""
    intent: {record['intent']}
    keywords: {record['keywords']}
    question: {record['question']}
    answer: {record['answer']}
    """
    
    texts.append(combined)

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Creating embeddings...")
embeddings = model.encode(texts).astype("float32")

faiss.normalize_L2(embeddings)
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

faiss.write_index(index, "faq.index")
with open("records.pkl", "wb") as f:
    pickle.dump(records, f)

print(f"✅ Index + records saved successfully! ({len(records)} records)")
