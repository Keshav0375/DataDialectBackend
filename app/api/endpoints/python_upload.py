from fastapi import APIRouter, HTTPException, File, UploadFile
from app.models.schemas import UploadResponse
from app.database.operations import get_upload_record, update_upload_record

router = APIRouter()

@router.post("/upload-python/{upload_id}", response_model=UploadResponse)
async def upload_python_file(upload_id: str, file: UploadFile = File(...)):
    """
    Upload Python file
    This completes the upload process by adding Python file to the existing record
    """
    try:
        existing_record = get_upload_record(upload_id)
        if not existing_record:
            raise HTTPException(status_code=404, detail="Upload ID not found")

        if not existing_record.get("credentials"):
            raise HTTPException(status_code=400, detail="Database credentials must be uploaded first")

        if not existing_record.get("csv_data"):
            raise HTTPException(status_code=400, detail="CSV file must be uploaded before Python file")

        if not file.filename.endswith('.py'):
            raise HTTPException(status_code=400, detail="Only Python files are allowed")

        contents = await file.read()

        try:
            python_content = contents.decode('utf-8')
            compile(python_content, file.filename, 'exec')
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="Invalid file encoding. Please use UTF-8")
        except SyntaxError as e:
            raise HTTPException(status_code=400, detail=f"Invalid Python syntax: {str(e)}")

        python_data = {
            "filename": file.filename,
            "content": python_content,
            "file_size": len(contents),
            "line_count": len(python_content.splitlines())
        }

        update_data = {
            "python_file": python_data,
            "status": "completed"
        }

        success = update_upload_record(upload_id, update_data)

        if success:
            return UploadResponse(
                success=True,
                message=f"Python file '{file.filename}' uploaded successfully. Upload process completed!",
                upload_id=upload_id
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to save Python file")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading Python file: {str(e)}")