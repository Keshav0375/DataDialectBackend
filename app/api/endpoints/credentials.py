from app.models.schemas import DatabaseCredentials, UploadResponse
from fastapi import APIRouter, HTTPException
from app.database.operations import create_new_upload_record, update_upload_record
from logging_config import setup_logger

router = APIRouter()
logger = setup_logger("DataDialect", "DataDialect.log")


@router.post("/upload-credentials", response_model=UploadResponse)
async def upload_database_credentials(credentials: DatabaseCredentials):
    """
    Upload Database Credentials and create first record in Database
    :returns upload_id
    """
    try:
        upload_id = create_new_upload_record()
        logger.info("Data Schema")
        credentials_data = {
            "credentials": credentials.dict(),
            "status": "csv_pending"
        }

        success = update_upload_record(upload_id, credentials_data)
        logger.info("Updating data and status of file")
        if success:
            return UploadResponse(
                success=True,
                message="Database credentials uploaded successfully",
                upload_id=upload_id
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to save credentials"
            )


    except Exception as E:
        raise HTTPException(status_code=500, detail=f"Error uploading credentials: {str(E)}")
