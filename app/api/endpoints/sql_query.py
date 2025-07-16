from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv

from app.nodes.SQL import SQLGraph
from app.models.schemas import UploadResponse, ChatMessage, SQLQueryRequest, SQLQueryResponse
from logging_config import setup_logger

load_dotenv()

router = APIRouter()
logger = setup_logger("DataDialect", "DataDialect.log")


def get_sql_graph():
    """Initialize SQLGraph with API credentials from environment variables"""
    try:
        api_key = os.getenv("API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        version = os.getenv("API_VERSION")
        openai_api_key = os.getenv("OPENAI_API_KEY")

        return SQLGraph(
            azure_api_key=api_key,
            azure_endpoint=endpoint,
            azure_api_version=version,
            openai_api_key=openai_api_key
        )
    except Exception as e:
        logger.error(f"Failed to initialize SQLGraph: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize SQL processing engine")


@router.post("/query-database/{upload_id}", response_model=SQLQueryResponse)
async def query_database(upload_id: str, request: SQLQueryRequest):
    """
    Query the database using natural language

    This endpoint processes natural language questions against uploaded database records
    and returns SQL query results along with natural language answers.

    Args:
        upload_id: The unique identifier for the uploaded database record
        request: SQLQueryRequest containing the question and chat history

    Returns:
        SQLQueryResponse with success status, answer, generated SQL query, and updated chat history
    """
    try:
        if upload_id != request.upload_id:
            raise HTTPException(
                status_code=400,
                detail="Upload ID in URL does not match request body"
            )

        messages_dict = []
        if request.messages:
            messages_dict = [{"role": msg.role, "content": msg.content} for msg in request.messages]

        sql_graph = get_sql_graph()

        result = sql_graph.invoke(
            upload_id=request.upload_id,
            question=request.question,
            messages=messages_dict
        )

        if result.get("success"):
            return SQLQueryResponse(
                success=True,
                answer=result.get("answer"),
                query=result.get("query"),
                upload_id=upload_id,
                messages=result.get("messages", [])
            )
        else:
            return SQLQueryResponse(
                success=False,
                error=result.get("error"),
                upload_id=upload_id
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing database query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing database query: {str(e)}"
        )
