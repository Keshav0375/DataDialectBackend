import os
from fastapi import APIRouter, HTTPException
from app.models.schemas import QueryExecutionResult, NoSQLState
from logging_config import setup_logger
from app.nodes.NoSQL import NoSQLGraph

logger = setup_logger("DataDialect", "DataDialect.log")
router = APIRouter()

@router.post("/query-execution", response_model=QueryExecutionResult)
async def execute_nosql_query(query_input: NoSQLState):
    """
    Execute natural language query against MongoDB collection using
    the generated schema and few-shot examples.
    """

    try:
        logger.info(f"Executing query: {query_input['question']}")
        azure_api_key = os.getenv("API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_version = os.getenv("API_VERSION", "2023-05-15")
        openai_api_key = os.getenv("OPENAI_API_KEY")

        nosql_graph = NoSQLGraph(
            azure_api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
            azure_api_version=azure_api_version,
            openai_api_key=openai_api_key
        )

        result = nosql_graph.invoke(
            mongo_uri=query_input["mongo_uri"],
            db_name=query_input["db_name"],
            collection_name=query_input["collection_name"],
            table_schema=query_input["table_schema"],
            schema_description=query_input["schema_description"],
            few_shot_examples=query_input["few_shot_examples"],
            question=query_input["question"],
            messages=query_input["messages"]
        )

        if result["success"]:
            logger.info(f"Query executed successfully, returned {result['result_count']} results")

            return QueryExecutionResult(
                success=True,
                question=result["question"],
                answer=result["answer"],
                query=result.get("query"),
                result_count=result["result_count"],
                response_type=result.get("response_type", "unknown"),
                execution_stats=result.get("execution_stats"),
                error=None
            )
        else:
            logger.error(f"Query execution failed: {result['error']}")
            raise HTTPException(
                status_code=400,
                detail=f"Query execution failed: {result['error']}"
            )


    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in query execution: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Query execution failed: {str(e)}"
        )

