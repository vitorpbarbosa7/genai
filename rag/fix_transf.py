
from dotenv import load_dotenv
import os
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

load_dotenv()

mongo_uri = os.getenv("MONGO_URI").replace("${DB_PASSWORD}", os.getenv("DB_PASSWORD"))
client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
db         = client["sample_mflix"]
collection = db["movies"]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def generate_embedding(text: str) -> list[float]:
    vec = model.encode(text)          # numpy.ndarray
    return vec.tolist()               # ← converte para lista de floats

# pega 20 docs que têm plot
for doc in collection.find({"plot": {"$exists": True}}).limit(20):
    emb = generate_embedding(doc["plot"])

    # grava só o campo novo, sem sobrescrever tudo
    collection.update_one(
        {"_id": doc["_id"]},
        {"$set": {"plot_embedding_hf": emb}},
        upsert=False
    )

print("✓ Embeddings gravados!")

