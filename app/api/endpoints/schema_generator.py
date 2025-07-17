from fastapi import APIRouter, HTTPException
from app.utils.mongo_analyser import MongoSchemaAnalyzer, SchemaAnalysisResult, NoSQLSchema
from app.models.schemas import NoSQLState
from logging_config import setup_logger

logger = setup_logger("DataDialect", "DataDialect.log")
router = APIRouter()
analyzer = MongoSchemaAnalyzer()

@router.post("/schema-creator", response_model=NoSQLState)
async def schema_input_analyser(schema_input: NoSQLSchema):
    """
        Analyze MongoDB collection schema and generate table schema,
        description, and few-shot examples for query generation.
        """
    try:
        logger.info(f"Starting schema analysis for collection: {schema_input.COLLECTION_NAME}")

        connection_stats = analyzer.test_mongodb_connection(
            schema_input.MONGO_URI,
            schema_input.DB_NAME,
            schema_input.COLLECTION_NAME
        )

        if not connection_stats["connection_success"]:
            raise HTTPException(
                status_code=400,
                detail=f"MongoDB connection failed: {connection_stats.get('error', 'Unknown error')}"
            )

        table_schema = analyzer.analyze_document_structure(schema_input.OBJECT)
        logger.info("Document structure analysis completed")

        schema_description = analyzer.generate_schema_description(
            table_schema,
            schema_input.OBJECT
        )
        logger.info("Schema description generated")

        few_shot_examples = analyzer.generate_few_shot_examples(
            table_schema,
            schema_input.COLLECTION_NAME,
            schema_input.OBJECT
        )
        logger.info(f"Generated {len(few_shot_examples)} few-shot examples")


        return NoSQLState(
            success=True,
            mongo_uri=schema_input.MONGO_URI,
            db_name=schema_input.DB_NAME,
            collection_name=schema_input.COLLECTION_NAME,
            table_schema=table_schema,
            schema_description=schema_description,
            few_shot_examples=few_shot_examples,
            collection_stats=connection_stats,
            question="",
            messages=[],
            result_count=0,
            collection=None,
            error=None,
            execution_stats=None,
            final_answer=None,
            generated_query=None,
            query_context=None,
            query_prompt_template=None,
            query_results=None,
            raw_query_response=None,
            response_type=None
        )


    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error in schema analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Schema analysis failed: {str(e)}"
        )


