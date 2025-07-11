from config import MONGODB_URL, DATABASE_NAME, COLLECTION_NAME
from pymongo import MongoClient

client = MongoClient(MONGODB_URL)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]
