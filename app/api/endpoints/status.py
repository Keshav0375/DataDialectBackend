from fastapi import APIRouter, HTTPException
from bson import ObjectId
from app.models.schemas import StatusResponse
from app.database.operations import get_upload_record
from app.database.connection import collection

router = APIRouter()


@router.get("/upload-status/{upload_id}", response_model=StatusResponse)
async def get_upload_status(upload_id: str):
    """
    Get the current status of an upload process
    """
    try:
        record = get_upload_record(upload_id)
        if not record:
            raise HTTPException(status_code=404, detail="Upload ID not found")

        return StatusResponse(
            upload_id=upload_id,
            has_credentials=record.get("credentials") is not None,
            has_csv=record.get("csv_data") is not None,
            has_python_file=record.get("python_file") is not None,
            created_at=record.get("created_at").isoformat(),
            updated_at=record.get("updated_at").isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching upload status: {str(e)}")


@router.get("/upload-details/{upload_id}")
async def get_upload_details(upload_id: str):
    """
    Get detailed information about an upload (excluding sensitive data)
    """
    try:
        record = get_upload_record(upload_id)
        if not record:
            raise HTTPException(status_code=404, detail="Upload ID not found")

        safe_record = {
            "upload_id": upload_id,
            "status": record.get("status"),
            "created_at": record.get("created_at").isoformat(),
            "updated_at": record.get("updated_at").isoformat(),
            "has_credentials": record.get("credentials") is not None,
            "has_csv": record.get("csv_data") is not None,
            "has_python_file": record.get("python_file") is not None
        }

        if record.get("csv_data"):
            safe_record["csv_info"] = {
                "filename": record["csv_data"]["filename"],
                "columns": record["csv_data"]["columns"],
                "row_count": record["csv_data"]["row_count"]
            }

        if record.get("python_file"):
            safe_record["python_info"] = {
                "filename": record["python_file"]["filename"],
                "file_size": record["python_file"]["file_size"],
                "line_count": record["python_file"]["line_count"]
            }

        return safe_record

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching upload details: {str(e)}")


@router.delete("/upload/{upload_id}")
async def delete_upload(upload_id: str):
    """
    Delete an upload record
    """
    try:
        object_id = ObjectId(upload_id)
        result = collection.delete_one({"_id": object_id})

        if result.deleted_count > 0:
            return {"success": True, "message": "Upload record deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Upload ID not found")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting upload: {str(e)}")