from pydantic import BaseModel, Field
from typing import List, Dict, TypedDict, Optional, Any
from langchain_community.utilities.sql_database import SQLDatabase

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

class SQLChatbot(BaseModel):
    upload_id: str
    query: str

class SQLState(TypedDict):
    upload_id: str
    question: str
    messages: List[Dict[str, str]]
    upload_record: Optional[Dict[str, Any]]
    table_info: Optional[str]
    relevant_tables: str
    generated_query: Optional[str]
    query_result: Optional[str]
    final_answer: Optional[str]
    error: Optional[str]
    cached_data: Dict[str, Any]
    relevant_examples: List[str]
    database_connection: Optional[SQLDatabase]
    database_uri: Optional[str]

class TableSchema(BaseModel):
    """Tables in SQL database"""
    name: str = Field(description="Name of table in SQL Database")


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class SQLQueryRequest(BaseModel):
    upload_id: str
    question: str
    messages: Optional[List[ChatMessage]] = []

class SQLQueryResponse(BaseModel):
    success: bool
    answer: Optional[str] = None
    query: Optional[str] = None
    upload_id: str
    messages: Optional[List[Dict[str, str]]] = None
    error: Optional[str] = None