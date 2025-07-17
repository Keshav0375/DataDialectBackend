from pydantic import BaseModel, Field
from typing import List, Dict, TypedDict, Optional, Any
from langchain_community.utilities.sql_database import SQLDatabase
from typing import Dict, List, Any, Optional
from typing_extensions import TypedDict

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


class NoSQLState(TypedDict):
    """State model for NoSQL MongoDB LangGraph workflow"""
    success: Optional[bool]
    question: str
    messages: List[Dict[str, str]]
    mongo_uri: str
    db_name: str
    collection_name: str
    table_schema: str
    schema_description: str
    few_shot_examples: List[Dict[str, Any]]
    collection: Optional[Any]
    collection_stats: Optional[Dict[str, Any]]
    query_prompt_template: Optional[str]
    generated_query: Optional[List[Dict[str, Any]]]
    raw_query_response: Optional[str]
    query_results: Optional[List[Dict[str, Any]]]
    result_count: int
    final_answer: Optional[str]
    error: Optional[str]
    query_context: Optional[Dict[str, Any]]
    execution_stats: Optional[Dict[str, Any]]
    response_type: Optional[str]


class NoSQLQueryRequest(BaseModel):
    """Request model for NoSQL MongoDB queries"""
    mongo_uri: str
    db_name: str
    collection_name: str
    table_schema: str
    schema_description: str
    few_shot_examples: List[Dict[str, Any]]
    question: str
    messages: Optional[List[Dict[str, str]]] = []


class NoSQLQueryResponse(BaseModel):
    """Response model for NoSQL MongoDB queries"""
    success: bool
    question: str
    answer: Optional[str] = None
    query: Optional[List[Dict[str, Any]]] = None
    raw_query: Optional[str] = None
    result_count: Optional[int] = 0
    response_type: Optional[str] = None
    execution_stats: Optional[Dict[str, Any]] = None
    messages: Optional[List[Dict[str, str]]] = []
    error: Optional[str] = None

class FewShotExample(BaseModel):
    """Few-shot example model"""
    input: str
    query: List[Dict[str, Any]]
    description: Optional[str] = None


class QueryExecutionResult(BaseModel):
    """Response model for query execution"""
    success: bool
    question: str
    answer: str
    query: Optional[List[Dict[str, Any]]]
    result_count: Optional[int]
    response_type: Optional[str]
    execution_stats: Optional[Dict[str, Any]]
    error: Optional[str] = None
