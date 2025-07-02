from dotenv import load_dotenv
import os
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# Load from .env
load_dotenv()
# Construct final URI
mongo_uri = os.getenv("MONGO_URI").replace("${DB_PASSWORD}", os.getenv("DB_PASSWORD"))
# Connect to MongoDB
client = MongoClient(mongo_uri)
# Example usage
db = client["sample_mflix"]
collection = db["movies"]

def generate_embedding(text:str) -> list[float]:
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(text)
    return embeddings

# para os 50 primeiros, gere embedding a partir dos plots
for doc in collection.find({'plot':{"$exists":True}}).limit(20):
    doc['plot_embedding_hf'] = generate_embedding(doc['plot'])
    collection.replace_one({'_id': doc['_id']}, doc)

