from dotenv import load_dotenv
import os
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI").replace("${DB_PASSWORD}",
                                                   os.getenv("DB_PASSWORD")))
collection = client["sample_mflix"]["movies"]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embed  = lambda text: model.encode(text).tolist()

query_text = "imaginary characters from outer space at war"

results = collection.aggregate([
    {
        "$vectorSearch": {
            "queryVector": embed(query_text),
            "path": "plot_embedding_hf",
            "numCandidates": 100,
            "limit": 4,
            "index": "KnnSearch"
        }
    }
])

for document in results:
    print(f'''Movie name: ' {document["title"]}, \nMovie Plot: {document["plot"]}\n''')
