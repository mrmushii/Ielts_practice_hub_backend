"""
MongoDB connection utility using the async Motor driver.
"""

from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os

load_dotenv()

class Database:
    client: AsyncIOMotorClient = None
    db = None

db = Database()

async def connect_to_mongo():
    """Connects to MongoDB and stores the client/db in the global Database structure."""
    uri = os.getenv("MONGODB_URI")
    if not uri:
        # Fallback to local MongoDB if URI is missing
        uri = "mongodb://localhost:27017"
        
    db.client = AsyncIOMotorClient(uri)
    db.db = db.client.ielts_platform
    print("Connected to MongoDB!")

async def close_mongo_connection():
    """Closes the MongoDB connection."""
    if db.client is not None:
        db.client.close()
        print("Closed MongoDB connection.")

def get_db():
    """Dependency to get the database instance."""
    return db.db
