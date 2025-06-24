from dotenv import load_dotenv
import os
from pymongo import MongoClient

# Load from .env
load_dotenv()

# Construct final URI
mongo_uri = os.getenv("MONGO_URI").replace("${DB_PASSWORD}", os.getenv("DB_PASSWORD"))

# Connect to MongoDB
client = MongoClient(mongo_uri)

# Example usage
db = client["sample_mflix"]
collection = db["movies"]

# Print collections to verify
print(db.list_collection_names())

