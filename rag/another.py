from dotenv import load_dotenv
import os
from pymongo import MongoClient

# Load environment variables from .env
load_dotenv()

# Extract environment variables
hf_token = os.getenv("HF_TOKEN")
db_password = os.getenv("DB_PASSWORD")
mongo_uri_template = os.getenv("MONGO_URI")

# Resolve password placeholder
if "${DB_PASSWORD}" in mongo_uri_template:
    mongo_uri = mongo_uri_template.replace("${DB_PASSWORD}", db_password)
else:
    mongo_uri = mongo_uri_template

# Debug print
print("[DEBUG] HF_TOKEN:", hf_token)
print("[DEBUG] DB_PASSWORD:", db_password)
print("[DEBUG] Raw MONGO_URI template:", mongo_uri_template)
print("[DEBUG] Final MONGO_URI:", mongo_uri)

# Try connecting to MongoDB
try:
    client = MongoClient(mongo_uri)
    db_names = client.list_database_names()  # This forces a connection
    print("[SUCCESS] Connected to MongoDB!")
    print("[INFO] Databases available:", db_names)
except Exception as e:
    print("[ERROR] Failed to connect to MongoDB:", e)
