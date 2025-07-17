from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import json
import os
from datetime import datetime

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pymongo import MongoClient
from logging_config import setup_logger

logger = setup_logger("DataDialect", "DataDialect.log")


class NoSQLSchema(BaseModel):
    """Enhanced input schema model for MongoDB analysis"""
    MONGO_URI: str = Field(..., description="MongoDB connection URI")
    DB_NAME: str = Field(..., description="Database name")
    COLLECTION_NAME: str = Field(..., description="Collection name")
    OBJECT: Dict = Field(..., description="Sample document from the collection")
    ANALYSIS_DEPTH: Optional[str] = Field("standard", description="Analysis depth: basic, standard, or detailed")


class NoSQLQuerySchema(BaseModel):
    """Input schema for MongoDB query execution"""
    MONGO_URI: str = Field(..., description="MongoDB connection URI")
    DB_NAME: str = Field(..., description="Database name")
    COLLECTION_NAME: str = Field(..., description="Collection name")
    TABLE_SCHEMA: str = Field(..., description="Generated table schema")
    SCHEMA_DESCRIPTION: str = Field(..., description="Schema description")
    FEW_SHOT_EXAMPLES: List[Dict[str, Any]] = Field(..., description="Few-shot examples")
    QUESTION: str = Field(..., description="User question in natural language")
    MESSAGES: Optional[List[Dict[str, str]]] = Field(default=[], description="Chat history")


class SchemaAnalysisResult(BaseModel):
    """Response model for schema analysis"""
    success: bool
    table_schema: str
    schema_description: str
    few_shot_examples: List[Dict[str, Any]]
    collection_stats: Optional[Dict[str, Any]]
    error: Optional[str]


# Initialize LLM for analysis
def get_analysis_llm():
    """Initialize LLM for schema analysis"""
    azure_api_key = os.getenv("API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_version = os.getenv("API_VERSION", "2023-05-15")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if azure_api_key and azure_endpoint:
        return AzureChatOpenAI(
            deployment_name="slideoo-chat-1",
            temperature=0.3,
            max_tokens=4000,
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version=azure_api_version,
        )
    elif openai_api_key:
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    else:
        raise ValueError("No valid API key provided for OpenAI or Azure")


class MongoSchemaAnalyzer:
    """Enhanced MongoDB schema analyzer with LLM integration"""

    def __init__(self):
        self.llm = get_analysis_llm()

    def analyze_document_structure(self, document: Dict) -> str:
        """Analyze document structure and generate schema"""

        def get_field_type(value, field_path=""):
            """Recursively determine field types"""
            if value is None:
                return "null"
            elif isinstance(value, bool):
                return "boolean"
            elif isinstance(value, int):
                return "integer"
            elif isinstance(value, float):
                return "number"
            elif isinstance(value, str):
                return "string"
            elif isinstance(value, datetime):
                return "date"
            elif isinstance(value, list):
                if len(value) == 0:
                    return "array"
                else:
                    # Analyze first element to determine array type
                    element_type = get_field_type(value[0], field_path)
                    return f"array<{element_type}>"
            elif isinstance(value, dict):
                # Recursively analyze nested objects
                nested_schema = {}
                for k, v in value.items():
                    nested_path = f"{field_path}.{k}" if field_path else k
                    nested_schema[k] = get_field_type(v, nested_path)
                return nested_schema
            else:
                return "mixed"

        schema = {}
        for key, value in document.items():
            schema[key] = get_field_type(value, key)

        return json.dumps(schema, indent=2)

    def generate_schema_description(self, schema: str, sample_doc: Dict) -> str:
        """Generate human-readable schema description using LLM"""
        prompt = PromptTemplate(
            template="""Analyze the following MongoDB document schema and provide a clear, structured description.

Schema Structure:
{schema}

Sample Document Keys: {doc_keys}

Generate a comprehensive schema description that includes:
1. Main purpose/domain of the collection
2. Key fields and their purposes
3. Data relationships and structure
4. Important patterns or conventions

Format the response as:
```
Schema Details:

1. field_name: Description of the field and its purpose
2. field_name: Description of the field and its purpose
...

Collection Purpose: Brief description of what this collection stores and its main use case.
```

Keep descriptions concise but informative. Focus on business meaning rather than just technical types.

Schema Description:""",
            input_variables=["schema", "doc_keys"]
        )

        chain = prompt | self.llm | StrOutputParser()

        try:
            description = chain.invoke({
                "schema": schema,
                "doc_keys": list(sample_doc.keys())
            })
            return description.strip()
        except Exception as e:
            logger.error(f"Error generating schema description: {e}")
            return f"Schema contains fields: {', '.join(sample_doc.keys())}"

    def generate_few_shot_examples(self, schema: str, collection_name: str, sample_doc: Dict) -> List[Dict[str, Any]]:
        """Generate few-shot examples using LLM"""
        prompt = PromptTemplate(
            template="""Generate 4-5 diverse MongoDB aggregation pipeline examples for the following schema.

Collection: {collection_name}
Schema: {schema}
Sample Fields: {sample_fields}

Create realistic query examples that demonstrate:
1. Simple filtering and projection
2. Aggregation with grouping
3. Complex filtering with multiple conditions
4. Counting/statistical operations
5. Sorting and limiting results

For each example, provide:
- "input": A natural language question
- "query": A valid MongoDB aggregation pipeline (as JSON array)

CRITICAL REQUIREMENTS:
- Return valid JSON format only
- Use actual field names from the schema
- Ensure all MongoDB operators are correct ($match, $group, $project, $sort, $limit, etc.)
- Use null instead of None
- Make queries realistic and useful
- Include proper aggregation patterns

Format as a JSON array:
[
    {{
        "input": "Natural language question",
        "query": [MongoDB aggregation pipeline stages]
    }},
    ...
]

Examples:""",
            input_variables=["collection_name", "schema", "sample_fields"]
        )

        chain = prompt | self.llm | StrOutputParser()

        try:
            response = chain.invoke({
                "collection_name": collection_name,
                "schema": schema,
                "sample_fields": list(sample_doc.keys())
            })

            # Clean and parse response
            cleaned_response = self._clean_json_response(response)
            examples = json.loads(cleaned_response)

            # Validate examples
            if isinstance(examples, list) and len(examples) > 0:
                return examples
            else:
                return self._generate_fallback_examples(sample_doc, collection_name)

        except Exception as e:
            logger.error(f"Error generating few-shot examples: {e}")
            return self._generate_fallback_examples(sample_doc, collection_name)


    def _clean_json_response(self, response: str) -> str:
        """Clean LLM response to valid JSON"""
        try:
            # Remove markdown code blocks
            response = response.replace("```json", "").replace("```", "")

            # Remove any text before the first [
            start_idx = response.find('[')
            if start_idx != -1:
                response = response[start_idx:]

            # Remove any text after the last ]
            end_idx = response.rfind(']')
            if end_idx != -1:
                response = response[:end_idx + 1]

            # Fix common JSON issues
            response = response.replace("'", '"')
            response = response.replace("None", "null")
            response = response.replace("True", "true")
            response = response.replace("False", "false")

            # Remove trailing commas before closing brackets
            import re
            response = re.sub(r',\s*}', '}', response)
            response = re.sub(r',\s*]', ']', response)

            # Test parse before returning
            json.loads(response)
            return response.strip()

        except json.JSONDecodeError as e:
            logger.error(f"JSON cleaning failed: {e}")
            # Return a basic valid JSON array
            return '[]'

    def _generate_fallback_examples(self, sample_doc: Dict, collection_name: str) -> List[Dict[str, Any]]:
        """Generate basic fallback examples when LLM fails"""
        sample_fields = list(sample_doc.keys())
        first_field = sample_fields[0] if sample_fields else "_id"

        return [
            {
                "input": f"Find all documents in {collection_name}",
                "query": [{"$limit": 100}]
            },
            {
                "input": f"Count total documents in {collection_name}",
                "query": [{"$count": "total_documents"}]
            },
            {
                "input": f"Get documents with specific {first_field}",
                "query": [
                    {"$match": {first_field: {"$exists": True}}},
                    {"$limit": 50}
                ]
            }
        ]

    def test_mongodb_connection(self, mongo_uri: str, db_name: str, collection_name: str) -> Dict[str, Any]:
        """Test MongoDB connection and get collection stats"""
        try:
            client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            client.server_info()  # Test connection

            db = client[db_name]
            collection = db[collection_name]

            # Get collection stats
            doc_count = collection.estimated_document_count()
            sample_doc = collection.find_one()

            client.close()

            return {
                "connection_success": True,
                "document_count": doc_count,
                "has_sample": sample_doc is not None,
                "sample_available": True
            }
        except Exception as e:
            logger.error(f"MongoDB connection test failed: {e}")
            return {
                "connection_success": False,
                "error": str(e),
                "document_count": 0,
                "has_sample": False
            }