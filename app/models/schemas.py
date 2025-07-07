from pydantic import BaseModel


class DatabaseCredentials(BaseModel):
    db_host: str
    db_user: str
    db_password: str
    db_name: str

class UploadResponse(BaseModel):
    success: bool
    message: str
    upload_id: str

class StatusResponse(BaseModel):
    upload_id: str
    has_credentials: bool
    has_csv: bool
    has_python_file: bool
    created_at: str
    updated_at: str