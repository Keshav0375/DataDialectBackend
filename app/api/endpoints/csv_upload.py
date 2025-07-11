from fastapi import APIRouter, HTTPException, File, UploadFile
import pandas as pd
import io
from app.database.operations import get_upload_record, update_upload_record
from app.models.schemas import UploadResponse

router = APIRouter()


@router.post("/upload-csv/{upload_id}", response_model=UploadResponse)
async def upload_csv_file(upload_id: str, file: UploadFile = File(...)):
    """
    Upload CSV file
    This adds CSV data to the existing record identified by upload_id
    """
    try:
        existing_record = get_upload_record(upload_id)
        if not existing_record:
            raise HTTPException(status_code=404, detail="Upload ID not found")

        if not existing_record.get("credentials"):
            raise HTTPException(status_code=400, detail="Database credentials must be uploaded first")

        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")

        contents = await file.read()

        try:
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
            csv_data = {
                "filename": file.filename,
                "columns": df.columns.tolist(),
                "row_count": len(df),
                "data": df.to_dict('records')[:100],
                "full_data_size": len(df)
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error parsing CSV file: {str(e)}")

        update_data = {
            "csv_data": csv_data,
            "status": "python_file_pending"
        }

        success = update_upload_record(upload_id, update_data)

        if success:
            return UploadResponse(
                success=True,
                message=f"CSV file '{file.filename}' uploaded successfully",
                upload_id=upload_id
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to save CSV file")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading CSV file: {str(e)}")