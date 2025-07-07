from datetime import datetime
from connection import collection
from bson import ObjectId
from logging_config import setup_logger

logger = setup_logger("DataDialect", "DataDialect.log")

def create_new_upload_record():
    """Create a new upload record in MongoDB"""
    record = {
        "credentials": None,
        "csv_data": None,
        "python_data": None,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "status": "credentials_pending"
    }
    result = collection.insert_one(record)
    return str(result.inserted_id)


def update_upload_record(upload_id: str, update_data: dict):
    """Update an existing upload record"""
    try:
        object_id = ObjectId(upload_id)
        update_data["updated_at"] = datetime.utcnow()
        result = collection.update_one(
            {
                "id": object_id
            },
            {
                "$set": update_data
            }
        )
        return result.modified_count > 0
    except Exception as E:
        logger.error(f"Error updating record: {E}")
        return False


def get_upload_record(upload_id: str):
    """Get an upload record by ID"""
    try:
        object_id = ObjectId(upload_id)
        return collection.find_one({
            "_id": object_id
        })
    except Exception as E:
        logger.error(f"Error fetching record: {E}")
        return None

