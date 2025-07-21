from typing import Dict, List, Any
import json
import re
import os
import random
from bson import ObjectId
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, AzureChatOpenAI, OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma

from langchain_aws import ChatBedrock
import boto3

from pymongo import MongoClient

from app.models.schemas import NoSQLState
from logging_config import setup_logger

logger = setup_logger("DataDialect", "DataDialect.log")

azure_api_key = os.getenv("API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_version = os.getenv("API_VERSION", "2023-05-15")
openai_api_key = os.getenv("OPENAI_API_KEY")

aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")


class CacheManager:
    def __init__(self):
        self.cache = {}

    def get(self, key: str) -> Any:
        return self.cache.get(key)

    def set(self, key: str, value: Any):
        self.cache[key] = value

    def clear(self):
        self.cache.clear()


cache_manager = CacheManager()


def serialize_for_msgpack(obj):
    """Convert MongoDB ObjectId and other non-serializable objects to strings"""
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: serialize_for_msgpack(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_msgpack(item) for item in obj]
    elif isinstance(obj, tuple):
        return [serialize_for_msgpack(item) for item in obj]
    elif hasattr(obj, '__dict__'):  # Handle custom objects
        return serialize_for_msgpack(obj.__dict__)
    else:
        try:
            # Test if it's JSON serializable
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            # If not serializable, convert to string
            return str(obj)


class MongoDBConnectionManager:
    """Manages MongoDB connections without storing them in state"""

    @staticmethod
    def create_connection(mongo_uri: str, db_name: str, collection_name: str):
        """Create and return MongoDB connection objects"""
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.server_info()  # Test connection

        db = client[db_name]
        collection = db[collection_name]

        return client, db, collection

    @staticmethod
    def get_collection_stats(collection):
        """Get collection statistics safely"""
        try:
            doc_count = collection.estimated_document_count()
            sample_doc = collection.find_one()
            return {
                "document_count": doc_count,
                "has_sample": sample_doc is not None
            }
        except Exception as e:
            logger.warning(f"Could not get collection stats: {e}")
            return {"document_count": 0, "has_sample": False}


class NoSQLNodes:
    def __init__(self, azure_api_key: str = None, azure_endpoint: str = None,
                 azure_api_version: str = None, openai_api_key: str = None,
                 aws_access_key_id: str = None,
                 aws_secret_access_key: str = None,
                 aws_region: str = "us-east-1",
                 claude_model_id: str = "anthropic.claude-3-5-sonnet-20240620-v1:0:0",
                 ):
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_region = aws_region
        self.claude_model_id = claude_model_id

        self.azure_api_key = azure_api_key
        self.azure_endpoint = azure_endpoint
        self.azure_api_version = azure_api_version
        self.openai_api_key = openai_api_key

        self.query_llm = self._initialize_query_llm()
        self.chat_llm = self._initialize_chat_llm()
        self.judge_llm = self._initialize_judge_llm()
        self.embeddings = self._initialize_embeddings()

    def _initialize_query_llm(self):
        """Initialize LLM for query generation with lower temperature - Priority: Claude > Azure > OpenAI"""

        # Try Claude via AWS Bedrock first
        try:
            if self.aws_access_key_id and self.aws_secret_access_key:
                logger.info("AWS Claude")
                return ChatBedrock(
                    model_id=self.claude_model_id,
                    region_name=self.aws_region,
                    credentials_profile_name=None,  # We'll set credentials directly
                    model_kwargs={
                        "max_tokens": 4000,
                        "temperature": 0.1,
                        "top_p": 0.9,
                    },
                    # Set AWS credentials
                    client=boto3.client(
                        'bedrock-runtime',
                        aws_access_key_id=self.aws_access_key_id,
                        aws_secret_access_key=self.aws_secret_access_key,
                        region_name=self.aws_region
                    )
                )
        except Exception as e:
            print(f"Failed to initialize Claude via Bedrock: {e}")

        # Fallback to Azure OpenAI
        try:
            if self.azure_api_key and self.azure_endpoint and self.azure_api_version:
                logger.info("Azure OpenAI")
                return AzureChatOpenAI(
                    deployment_name="slideoo-chat-1",
                    temperature=0.1,
                    max_tokens=4000,
                    azure_endpoint=self.azure_endpoint,
                    api_key=self.azure_api_key,
                    api_version=self.azure_api_version,
                )
        except Exception as e:
            print(f"Failed to initialize Azure OpenAI: {e}")

        # Final fallback to OpenAI
        try:
            if self.openai_api_key:
                logger.info("OpenAI")
                return ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.1,
                    max_tokens=4000,
                    api_key=self.openai_api_key
                )
        except Exception as e:
            print(f"Failed to initialize OpenAI: {e}")

        raise ValueError("No valid API credentials provided for Claude, Azure OpenAI, or OpenAI")

    def _initialize_chat_llm(self):
        """Initialize LLM for conversational responses with higher temperature - Priority: Claude > Azure > OpenAI"""

        # Try Claude via AWS Bedrock first
        try:
            if self.aws_access_key_id and self.aws_secret_access_key:
                return ChatBedrock(
                    model_id=self.claude_model_id,
                    region_name=self.aws_region,
                    credentials_profile_name=None,
                    model_kwargs={
                        "max_tokens": 2000,
                        "temperature": 0.7,
                        "top_p": 0.9,
                    },
                    client=boto3.client(
                        'bedrock-runtime',
                        aws_access_key_id=self.aws_access_key_id,
                        aws_secret_access_key=self.aws_secret_access_key,
                        region_name=self.aws_region
                    )
                )
        except Exception as e:
            print(f"Failed to initialize Claude via Bedrock for chat: {e}")

        # Fallback to Azure OpenAI
        try:
            if self.azure_api_key and self.azure_endpoint and self.azure_api_version:
                return AzureChatOpenAI(
                    deployment_name="slideoo-chat-1",
                    temperature=0.7,
                    max_tokens=2000,
                    azure_endpoint=self.azure_endpoint,
                    api_key=self.azure_api_key,
                    api_version=self.azure_api_version,
                )
        except Exception as e:
            print(f"Failed to initialize Azure OpenAI for chat: {e}")

        # Final fallback to OpenAI
        try:
            if self.openai_api_key:
                return ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.7,
                    max_tokens=2000,
                    api_key=self.openai_api_key
                )
        except Exception as e:
            print(f"Failed to initialize OpenAI for chat: {e}")

        raise ValueError("No valid API credentials provided for Claude, Azure OpenAI, or OpenAI")

    def _initialize_judge_llm(self):
        """Initialize LLM for judging response quality with very low temperature"""

        # Try Claude via AWS Bedrock first
        try:
            if self.aws_access_key_id and self.aws_secret_access_key:
                return ChatBedrock(
                    model_id=self.claude_model_id,
                    region_name=self.aws_region,
                    credentials_profile_name=None,
                    model_kwargs={
                        "max_tokens": 1000,
                        "temperature": 0.0,
                        "top_p": 0.8,
                    },
                    client=boto3.client(
                        'bedrock-runtime',
                        aws_access_key_id=self.aws_access_key_id,
                        aws_secret_access_key=self.aws_secret_access_key,
                        region_name=self.aws_region
                    )
                )
        except Exception as e:
            print(f"Failed to initialize Claude via Bedrock for judge: {e}")

        # Fallback to Azure OpenAI
        try:
            if self.azure_api_key and self.azure_endpoint and self.azure_api_version:
                return AzureChatOpenAI(
                    deployment_name="slideoo-chat-1",
                    temperature=0.0,
                    max_tokens=1000,
                    azure_endpoint=self.azure_endpoint,
                    api_key=self.azure_api_key,
                    api_version=self.azure_api_version,
                )
        except Exception as e:
            print(f"Failed to initialize Azure OpenAI for judge: {e}")

        # Final fallback to OpenAI
        try:
            if self.openai_api_key:
                return ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.0,
                    max_tokens=1000,
                    api_key=self.openai_api_key
                )
        except Exception as e:
            print(f"Failed to initialize OpenAI for judge: {e}")

        raise ValueError("No valid API credentials provided for Claude, Azure OpenAI, or OpenAI")

    def _initialize_embeddings(self):
        """Initialize embeddings for example selection"""
        if self.azure_api_key and self.azure_endpoint:
            return AzureOpenAIEmbeddings(
                azure_deployment="Text-Analytics",
                api_key=self.azure_api_key,
                azure_endpoint=self.azure_endpoint,
                api_version=self.azure_api_version
            )
        elif self.openai_api_key:
            return OpenAIEmbeddings(api_key=self.openai_api_key)
        return None

    def initialize_mongodb_connection_node(self, state: NoSQLState) -> NoSQLState:
        """Node 1: Test MongoDB connection and store connection info (not objects)"""
        logger.info("Initializing MongoDB connection")

        try:
            mongo_uri = state["mongo_uri"]
            db_name = state["db_name"]
            collection_name = state["collection_name"]

            if not mongo_uri:
                raise ValueError("MongoDB URI is required")

            # Test connection
            client, db, collection = MongoDBConnectionManager.create_connection(
                mongo_uri, db_name, collection_name
            )

            # Get collection stats
            collection_stats = MongoDBConnectionManager.get_collection_stats(collection)

            # Store connection info, not the objects themselves
            state["collection_stats"] = collection_stats

            # Close the test connection
            logger.info(
                f"MongoDB connection successful - {collection_stats['document_count']} documents in {collection_name}")
            client.close()

        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            state["error"] = f"MongoDB connection failed: {str(e)}"

        return state

    def create_optimized_query_prompt_node(self, state: NoSQLState) -> NoSQLState:
        """Node 2: Create optimized query generation prompt with provided schema and examples"""
        logger.info("Creating optimized query generation prompt")

        try:
            # Use provided schema info
            table_schema = state.get("table_schema", "")
            schema_description = state.get("schema_description", "")
            few_shot_examples = state.get("few_shot_examples", [])

            if not table_schema:
                raise ValueError("Table schema is required")

            # Create enhanced prompt template
            query_prompt_template = """
You are an expert MongoDB query specialist. Create MongoDB aggregation pipelines based on user questions.

CRITICAL RULES:
1. Return ONLY a valid JSON array containing the MongoDB pipeline stages
2. Use null instead of None for null values
3. Do not include any wrapper text, explanations, or markdown
4. Ensure all field names use underscores, not asterisks or other characters
5. Use proper MongoDB operators like $match, $group, $project, $sort, $limit, $lookup, $unwind
6. For text search, use $regex with case-insensitive options
7. For counting, use $group with $sum: 1
8. For date operations, use appropriate MongoDB date operators
9. Always validate field names against the schema
10. Use $limit wisely to prevent excessive data retrieval

Collection Schema:
{table_schema}

Schema Description:
{schema_description}

Few-Shot Examples:
{examples}

User Question: {user_question}

Return only the MongoDB aggregation pipeline as a JSON array:
"""

            state["query_prompt_template"] = query_prompt_template
            logger.info("Query prompt template created successfully")

        except Exception as e:
            logger.error(f"Error creating query prompt: {e}")
            state["error"] = f"Prompt creation failed: {str(e)}"

        return state

    def intelligent_query_generation_node(self, state: NoSQLState) -> NoSQLState:
        """Node 3: Generate MongoDB query with embeddings-based example selection"""
        logger.info("Generating MongoDB aggregation pipeline")

        try:
            question = state["question"]
            table_schema = state.get("table_schema", "")
            schema_description = state.get("schema_description", "")
            few_shot_examples = state.get("few_shot_examples", [])

            # Use learning examples if available from fallback flow
            if state.get("learning_examples"):
                few_shot_examples.extend(state["learning_examples"])
                logger.info(f"Using {len(state['learning_examples'])} learning examples")

            # Use rephrased question if available
            effective_question = state.get("rephrased_question", question)

            # Select most relevant examples using embeddings
            relevant_examples = self._select_relevant_examples(effective_question, few_shot_examples)

            # Create query prompt
            query_prompt = PromptTemplate(
                template=state["query_prompt_template"],
                input_variables=["table_schema", "schema_description", "examples", "user_question"]
            )

            # Generate query
            chain = query_prompt | self.query_llm | StrOutputParser()

            response = chain.invoke({
                "table_schema": table_schema,
                "schema_description": schema_description,
                "examples": json.dumps(relevant_examples, indent=2),
                "user_question": effective_question
            })

            cleaned_response = self._clean_query_response(response)
            pipeline = self._validate_and_parse_json(cleaned_response)

            # Add safety limits
            # After pipeline generation and before setting state
            pipeline = self._add_safety_limits(pipeline)

            # Validate the query
            if not self._validate_mongodb_query(pipeline):
                logger.warning("Generated query failed validation, using simple fallback")
                pipeline = [{"$match": {}}, {"$limit": 10}]


            state["generated_query"] = pipeline
            state["raw_query_response"] = response
            state["query_context"] = {
                "examples_used": len(relevant_examples),
                "pipeline_stages": len(pipeline),
                "used_learning": bool(state.get("learning_examples")),
                "used_rephrased": bool(state.get("rephrased_question"))
            }

            logger.info(f"Generated {len(pipeline)} stage pipeline")

        except Exception as e:
            logger.error(f"Error generating MongoDB query: {e}")
            state["error"] = f"Query generation failed: {str(e)}"

        return state

    def execute_mongodb_query_node(self, state: NoSQLState) -> NoSQLState:
        """Node 4: Execute MongoDB query with fresh connection"""
        logger.info("Executing MongoDB aggregation pipeline")

        try:
            pipeline = state["generated_query"]
            if not pipeline:
                raise ValueError("No query pipeline generated")

            # Create fresh connection for execution
            mongo_uri = state["mongo_uri"]
            db_name = state["db_name"]
            collection_name = state["collection_name"]

            client, db, collection = MongoDBConnectionManager.create_connection(
                mongo_uri, db_name, collection_name
            )

            try:
                # Execute with timeout
                results = list(collection.aggregate(pipeline, maxTimeMS=30000))

                # Serialize results
                serialized_results = serialize_for_msgpack(results)

                state["query_results"] = serialized_results
                state["result_count"] = len(results)
                state["execution_stats"] = {
                    "pipeline_stages": len(pipeline),
                    "documents_returned": len(results)
                }

                logger.info(f"Query executed successfully - {len(results)} documents returned")

            finally:
                # Always close the connection
                client.close()

        except Exception as e:
            logger.error(f"Error executing MongoDB query: {e}")
            state["error"] = f"Query execution failed: {str(e)}"

        return state

    def adaptive_conversational_response_node(self, state: NoSQLState) -> NoSQLState:
        """Node 5: Generate adaptive conversational response based on result complexity"""
        logger.info("Generating adaptive conversational response")

        try:
            question = state["question"]

            # Handle execution errors
            if state.get("error") and "Query execution failed" in state.get("error", ""):
                response = f"I encountered an issue with the database query. This might be due to incorrect field usage or query structure. Let me try a different approach."
                state["final_answer"] = response
                state["response_type"] = "execution_error"
                return state

            results = state["query_results"]
            result_count = state["result_count"]

            # Rest of the existing logic...
            if result_count == 0:
                response = self._generate_no_results_response(question)
            elif result_count <= 5:
                response = self._generate_detailed_response(question, results, result_count)
            elif result_count <= 50:
                response = self._generate_summary_response(question, results, result_count)
            else:
                response = self._generate_high_level_response(question, results, result_count)

            state["final_answer"] = response
            state["response_type"] = self._get_response_type(result_count)

            logger.info(f"Generated {state['response_type']} response for {result_count} results")

        except Exception as e:
            logger.error(f"Error generating conversational response: {e}")
            # Fallback response
            state["final_answer"] = self._generate_fallback_response(
                state["question"],
                state.get("result_count", 0),
                state.get("query_results", [])
            )
            state["response_type"] = "fallback"

        return state

    def response_quality_judge_node(self, state: NoSQLState) -> NoSQLState:
        """Node 6: Judge the quality of the generated response"""
        logger.info("Judging response quality")

        try:
            question = state["question"]
            answer = state.get("final_answer", "")
            result_count = state.get("result_count", 0)
            query = state.get("generated_query", [])
            response_type = state.get("response_type", "")

            # Skip judging if already in fallback mode
            if state.get("fallback_attempt", False):
                state["quality_score"] = 10  # Accept fallback attempts
                logger.info("Skipping judge for fallback attempt")
                return state

            # Automatic low score for execution errors to trigger fallback
            if response_type == "execution_error" or "Query execution failed" in answer:
                state["quality_score"] = 2
                state["judge_reasoning"] = "Query execution error detected - fallback needed"
                logger.info("Auto-assigned low quality score for execution error")
                return state

            # Rest of the existing judge logic...
            judge_prompt = PromptTemplate(
                template="""
    You are a quality judge for database query responses. Evaluate the response quality on a scale of 1-10.

    Question: {question}
    Generated MongoDB Query: {query}
    Result Count: {result_count}
    Response: {answer}
    Response Type: {response_type}

    Quality Criteria:
    1. Relevance: Does the response directly answer the question? (0-3 points)
    2. Completeness: Is the answer complete and informative? (0-3 points)
    3. Accuracy: Does the response make sense given the query results? (0-2 points)
    4. Clarity: Is the response clear and well-formatted? (0-2 points)

    Consider these as LOW QUALITY (score 1-4):
    - Query execution errors or technical failures
    - Generic responses that don't address the specific question
    - Empty or very vague answers
    - Responses that ignore available data
    - Technical errors or inconsistencies

    Return ONLY a number from 1-10:
    """,
                input_variables=["question", "query", "result_count", "answer", "response_type"]
            )

            chain = judge_prompt | self.judge_llm | StrOutputParser()

            response = chain.invoke({
                "question": question,
                "query": json.dumps(query, default=str),
                "result_count": result_count,
                "answer": answer,
                "response_type": response_type
            })

            # Extract numeric score
            score_match = re.search(r'\b(\d+)\b', response.strip())
            quality_score = int(score_match.group(1)) if score_match else 5

            state["quality_score"] = quality_score
            state["judge_reasoning"] = response

            logger.info(f"Quality score: {quality_score}/10")

        except Exception as e:
            logger.error(f"Error in quality judgment: {e}")
            state["quality_score"] = 3  # Low score to trigger fallback on judge errors
            state["judge_reasoning"] = f"Judge error: {str(e)}"

        return state

    def _create_validated_learning_examples(self, documents: List[Dict],
                                            field_analysis: Dict,
                                            exploratory_results: Dict) -> List[Dict]:
        """Create learning examples based on validated database exploration"""
        examples = []

        # Example 1: Field existence (validated)
        field_existence = exploratory_results.get("field_existence", {})
        for field, count in field_existence.items():
            if count > 0:
                examples.append({
                    "input": f"Find documents that have {field}",
                    "query": [{"$match": {field: {"$exists": True}}}, {"$limit": 20}],
                    "description": f"Field existence check for {field} (validated: {count} docs)",
                    "validation": f"Known to work - {count} documents have this field"
                })

        # Example 2: Text search (validated with real patterns)
        data_patterns = exploratory_results.get("data_patterns", {})
        for field, sample_values in data_patterns.items():
            if sample_values:
                examples.append({
                    "input": f"Search for text in {field}",
                    "query": [{"$match": {field: {"$regex": ".*search_term.*", "$options": "i"}}}, {"$limit": 20}],
                    "description": f"Text search in {field}",
                    "validation": f"Field contains text values like: {sample_values[:2]}"
                })

        # Example 3: Array operations (validated)
        array_fields = exploratory_results.get("array_fields", {})
        for field, field_info in array_fields.items():
            if field_info.get("has_arrays"):
                examples.append({
                    "input": f"Find documents where {field} array has more than 2 elements",
                    "query": [
                        {"$match": {field: {"$exists": True, "$type": "array"}}},
                        {"$match": {f"{field}.2": {"$exists": True}}},  # Safe way to check array size
                        {"$limit": 20}
                    ],
                    "description": f"Array size check for {field}",
                    "validation": "Validated array field existence and safe size checking"
                })

        # Example 4: Counting (always works)
        examples.append({
            "input": "How many documents are there?",
            "query": [{"$count": "total"}],
            "description": "Count all documents",
            "validation": "Universal counting query"
        })

        # Example 5: Aggregation with grouping (safe approach)
        if field_existence:
            first_field = list(field_existence.keys())[0]
            examples.append({
                "input": f"Group by {first_field}",
                "query": [
                    {"$match": {first_field: {"$exists": True}}},
                    {"$group": {"_id": f"${first_field}", "count": {"$sum": 1}}},
                    {"$sort": {"count": -1}},
                    {"$limit": 20}
                ],
                "description": f"Grouping by {first_field}",
                "validation": f"Safe grouping - field exists in {field_existence[first_field]} documents"
            })

        # Example 6: Date operations (if date fields found)
        date_fields = exploratory_results.get("date_fields", {})
        for field, date_info in date_fields.items():
            if date_info.get("count", 0) > 0:
                examples.append({
                    "input": f"Find recent documents by {field}",
                    "query": [
                        {"$match": {field: {"$exists": True}}},
                        {"$sort": {field: -1}},
                        {"$limit": 20}
                    ],
                    "description": f"Date sorting by {field}",
                    "validation": f"Validated date field with {date_info.get('count', 0)} documents"
                })

        logger.info(f"Created {len(examples)} validated learning examples")
        return examples[:8]  # Limit to prevent token overflow


    def database_learning_node(self, state: NoSQLState) -> NoSQLState:
        """Node 7: Learn database patterns through exploratory queries"""
        logger.info("Learning database structure through pattern analysis and validation")

        try:
            mongo_uri = state["mongo_uri"]
            db_name = state["db_name"]
            collection_name = state["collection_name"]

            client, db, collection = MongoDBConnectionManager.create_connection(
                mongo_uri, db_name, collection_name
            )

            try:
                # Step 1: Get initial samples
                total_docs = collection.estimated_document_count()
                sample_size = min(10, max(5, total_docs // 1000))
                sample_pipeline = [{"$sample": {"size": sample_size}}]
                sample_docs = list(collection.aggregate(sample_pipeline))

                # Step 2: Analyze patterns from samples
                field_analysis = self._analyze_document_structure(sample_docs)

                # Step 3: Generate and execute exploratory queries
                exploratory_results = self._execute_exploratory_queries(collection, field_analysis)

                # Step 4: Create validated learning examples
                learning_examples = self._create_validated_learning_examples(
                    sample_docs, field_analysis, exploratory_results
                )

                state["sample_documents"] = serialize_for_msgpack(sample_docs)
                state["field_analysis"] = serialize_for_msgpack(field_analysis)
                state["exploratory_results"] = serialize_for_msgpack(exploratory_results)
                state["learning_examples"] = serialize_for_msgpack(learning_examples)

                logger.info(f"Learned from {len(sample_docs)} samples + {len(exploratory_results)} exploratory queries")

            finally:
                client.close()

        except Exception as e:
            logger.error(f"Error in database learning: {e}")
            state["sample_documents"] = []
            state["field_analysis"] = {}
            state["exploratory_results"] = {}
            state["learning_examples"] = []

        return state

    def database_learning_node(self, state: NoSQLState) -> NoSQLState:
        """Node 7: Learn database patterns through exploratory queries"""
        logger.info("Learning database structure through pattern analysis and validation")

        try:
            mongo_uri = state["mongo_uri"]
            db_name = state["db_name"]
            collection_name = state["collection_name"]

            client, db, collection = MongoDBConnectionManager.create_connection(
                mongo_uri, db_name, collection_name
            )

            try:
                # Step 1: Get initial samples
                total_docs = collection.estimated_document_count()
                sample_size = min(10, max(5, total_docs // 1000))
                sample_pipeline = [{"$sample": {"size": sample_size}}]
                sample_docs = list(collection.aggregate(sample_pipeline))

                # Step 2: Analyze patterns from samples
                field_analysis = self._analyze_document_structure(sample_docs)

                # Step 3: Generate and execute exploratory queries
                exploratory_results = self._execute_exploratory_queries(collection, field_analysis)

                # Step 4: Create validated learning examples
                learning_examples = self._create_validated_learning_examples(
                    sample_docs, field_analysis, exploratory_results
                )

                state["sample_documents"] = serialize_for_msgpack(sample_docs)
                state["field_analysis"] = field_analysis
                state["exploratory_results"] = exploratory_results
                state["learning_examples"] = learning_examples

                logger.info(f"Learned from {len(sample_docs)} samples + {len(exploratory_results)} exploratory queries")

            finally:
                client.close()

        except Exception as e:
            logger.error(f"Error in database learning: {e}")
            state["sample_documents"] = []
            state["field_analysis"] = {}
            state["exploratory_results"] = {}
            state["learning_examples"] = []

        return state

    def question_rephrase_node(self, state: NoSQLState) -> NoSQLState:
        """Node 8: Rephrase question based on validated database understanding"""
        logger.info("Rephrasing question based on validated database understanding")

        try:
            original_question = state["question"]
            field_analysis = state.get("field_analysis", {})
            exploratory_results = state.get("exploratory_results", {})

            # Get validated field information
            field_existence = exploratory_results.get("field_existence", {})
            data_patterns = exploratory_results.get("data_patterns", {})

            rephrase_prompt = PromptTemplate(
                template="""
    You are a database expert. Rephrase the user question based on VALIDATED database exploration results.

    Original Question: {original_question}

    VALIDATED Field Information:
    {field_existence}

    VALIDATED Data Patterns:
    {data_patterns}

    Available Array Fields:
    {array_fields}

    Date Fields:
    {date_fields}

    Instructions:
    1. Use ONLY field names that are validated to exist in the database
    2. Make the question more specific using actual field names and patterns
    3. Consider the data types and structures that were validated through exploration
    4. Avoid operations that might fail (like incorrect $size usage on non-arrays)
    5. If original question mentions arrays, use safe array operations
    6. Use field names exactly as they appear in the validation results

    Return only the rephrased question that will work with this specific database:
    """,
                input_variables=["original_question", "field_existence", "data_patterns", "array_fields", "date_fields"]
            )

            chain = rephrase_prompt | self.query_llm | StrOutputParser()

            rephrased = chain.invoke({
                "original_question": original_question,
                "field_existence": json.dumps(field_existence, indent=2),
                "data_patterns": json.dumps(data_patterns, indent=2),
                "array_fields": json.dumps(exploratory_results.get("array_fields", {}), indent=2),
                "date_fields": json.dumps(exploratory_results.get("date_fields", {}), indent=2)
            })

            state["rephrased_question"] = rephrased.strip()
            logger.info(f"Rephrased with validation: {rephrased.strip()}")

        except Exception as e:
            logger.error(f"Error rephrasing question: {e}")
            state["rephrased_question"] = state["question"]

        return state

    def _analyze_document_structure(self, documents: List[Dict]) -> Dict:
        """Analyze the structure of sample documents"""
        field_types = {}
        field_examples = {}
        nested_fields = {}

        for doc in documents:
            self._analyze_single_document(doc, field_types, field_examples, nested_fields)

        return {
            "field_types": field_types,
            "field_examples": field_examples,
            "nested_fields": nested_fields,
            "total_unique_fields": len(field_types)
        }

    def _analyze_single_document(self, doc: Dict, field_types: Dict, field_examples: Dict, nested_fields: Dict,
                                 prefix: str = ""):
        """Recursively analyze a single document structure"""
        for key, value in doc.items():
            full_key = f"{prefix}{key}" if prefix else key

            # Skip MongoDB internal fields
            if key.startswith('_') and key != '_id':
                continue

            # Determine type
            value_type = type(value).__name__
            if value_type not in field_types.get(full_key, []):
                field_types.setdefault(full_key, []).append(value_type)

            # Store example (limit string length)
            if full_key not in field_examples:
                if isinstance(value, str):
                    field_examples[full_key] = value[:100] + "..." if len(str(value)) > 100 else value
                elif isinstance(value, (int, float, bool)):
                    field_examples[full_key] = value
                elif isinstance(value, list) and len(value) > 0:
                    field_examples[full_key] = f"Array of {type(value[0]).__name__}, length: {len(value)}"

            # Handle nested objects
            if isinstance(value, dict) and len(value) < 20:  # Avoid too deep nesting
                nested_fields[full_key] = list(value.keys())
                self._analyze_single_document(value, field_types, field_examples, nested_fields, f"{full_key}.")

    def _create_learning_examples(self, documents: List[Dict], field_analysis: Dict) -> List[Dict]:
        """Create learning examples from sample documents"""
        examples = []

        # Create basic field access examples
        if field_analysis.get("field_examples"):
            for field, example in list(field_analysis["field_examples"].items())[:5]:
                if not field.startswith('_'):
                    examples.append({
                        "input": f"Find documents with {field}",
                        "query": [{"$match": {field: {"$exists": True}}}, {"$limit": 10}],
                        "description": f"Basic field access for {field}"
                    })

        # Create counting examples
        if documents:
            examples.append({
                "input": "How many documents are there?",
                "query": [{"$count": "total"}],
                "description": "Count all documents"
            })

        # Create field-specific examples based on data types
        for field, types in field_analysis.get("field_types", {}).items():
            if "str" in types and not field.startswith('_'):
                examples.append({
                    "input": f"Search for text in {field}",
                    "query": [{"$match": {field: {"$regex": "search_term", "$options": "i"}}}, {"$limit": 10}],
                    "description": f"Text search in {field}"
                })
                break  # Just one text search example

        return examples[:7]  # Limit to prevent token overflow

    def _select_relevant_examples(self, question: str, examples: List[Dict]) -> List[Dict]:
        """Select most relevant examples using embeddings"""
        if not examples:
            return []

        # Try embeddings first, but with better error handling
        if self.embeddings:
            try:
                # Filter out complex metadata from examples for embeddings
                simplified_examples = []
                for example in examples:
                    simplified_example = {
                        "input": example.get("input", ""),
                        "query": json.dumps(example.get("query", []))  # Convert to string
                    }
                    simplified_examples.append(simplified_example)

                example_selector = SemanticSimilarityExampleSelector.from_examples(
                    simplified_examples,
                    self.embeddings,
                    Chroma,
                    k=min(3, len(simplified_examples)),
                    input_keys=["input"],
                )
                selected = example_selector.select_examples({"input": question})

                # Convert back to original format
                result = []
                for selected_example in selected:
                    # Find the original example
                    for original in examples:
                        if original.get("input") == selected_example.get("input"):
                            result.append(original)
                            break

                return result

            except Exception as e:
                logger.warning(f"Embedding selection failed: {e}")

        # If embeddings fail, return first 3 examples
        return examples[:3]

    def _execute_exploratory_queries(self, collection, field_analysis: Dict) -> Dict:
        """Execute exploratory queries to validate and expand database understanding"""
        exploratory_results = {}

        try:
            # Query 1: Field existence and counts
            field_existence = {}
            for field in list(field_analysis.get("field_types", {}).keys())[:5]:
                if not field.startswith('_'):
                    try:
                        count = collection.count_documents({field: {"$exists": True}})
                        field_existence[field] = count
                    except:
                        field_existence[field] = 0

            exploratory_results["field_existence"] = field_existence

            # Query 2: Data type validation for key fields
            data_patterns = {}
            for field, types in field_analysis.get("field_types", {}).items():
                if "str" in types and not field.startswith('_'):
                    try:
                        # Get sample values to understand patterns
                        sample_values = list(collection.aggregate([
                            {"$match": {field: {"$exists": True, "$type": "string"}}},
                            {"$project": {field: 1}},
                            {"$limit": 5}
                        ]))
                        data_patterns[field] = [doc.get(field, "") for doc in sample_values]
                    except:
                        data_patterns[field] = []
                    break  # Just one to avoid too many queries

            exploratory_results["data_patterns"] = data_patterns

            # Query 3: Array field analysis
            array_fields = {}
            for field, types in field_analysis.get("field_types", {}).items():
                if "list" in types and not field.startswith('_'):
                    try:
                        # Check array sizes and sample elements
                        array_sample = list(collection.aggregate([
                            {"$match": {field: {"$exists": True, "$type": "array"}}},
                            {"$project": {
                                field: 1,
                                f"{field}_size": {"$size": f"${field}"}
                            }},
                            {"$limit": 3}
                        ]))
                        array_fields[field] = {
                            "samples": array_sample,
                            "has_arrays": len(array_sample) > 0
                        }
                    except Exception as e:
                        logger.warning(f"Array analysis failed for {field}: {e}")
                        array_fields[field] = {"samples": [], "has_arrays": False}
                    break  # Just one array field to avoid complexity

            exploratory_results["array_fields"] = array_fields

            # Query 4: Date field analysis (if any)
            date_fields = {}
            for field, types in field_analysis.get("field_types", {}).items():
                if "datetime" in types or "date" in field.lower():
                    try:
                        date_range = list(collection.aggregate([
                            {"$match": {field: {"$exists": True}}},
                            {"$group": {
                                "_id": None,
                                "min_date": {"$min": f"${field}"},
                                "max_date": {"$max": f"${field}"},
                                "count": {"$sum": 1}
                            }}
                        ]))
                        date_fields[field] = date_range[0] if date_range else {}
                    except:
                        date_fields[field] = {}
                    break  # Just one date field

            exploratory_results["date_fields"] = date_fields

            logger.info(f"Completed {len(exploratory_results)} exploratory query types")

        except Exception as e:
            logger.error(f"Exploratory queries failed: {e}")

        return serialize_for_msgpack(exploratory_results)

    def _add_safety_limits(self, pipeline: List[Dict]) -> List[Dict]:
        """Add safety limits to prevent excessive data retrieval with enhanced token management"""
        # Check if pipeline already has limit or group operations
        has_limit = any("$limit" in stage for stage in pipeline)
        has_group = any("$group" in stage for stage in pipeline)
        has_count = any("$count" in stage for stage in pipeline)

        # Enhanced safety limits based on operation type
        if not has_limit and not has_group and not has_count:
            # For regular queries, use smaller limit to prevent token overflow
            pipeline.append({"$limit": 100})
        elif has_group and not has_limit:
            # For aggregation queries, add limit after grouping
            pipeline.append({"$limit": 50})

        # Add projection to limit field size if documents are large
        if not any("$project" in stage for stage in pipeline):
            # Add basic projection to exclude very large fields
            pipeline.insert(-1 if has_limit else len(pipeline), {
                "$project": {
                    "_id": 1,
                    # Add common fields that are usually not too large
                    **{field: 1 for field in
                       ["name", "title", "status", "date", "created_at", "updated_at", "type", "category"][:5]}
                }
            })

        return pipeline

    def _generate_detailed_response(self, question: str, results: List[Dict], count: int) -> str:
        """Generate detailed response for small result sets with token management"""
        try:
            # Limit result size for token management
            limited_results = results[:3] if count > 3 else results

            prompt = PromptTemplate(
                template="""Analyze these MongoDB query results and provide a concise, direct response.

Question: {question}
Results ({count} documents, showing first {shown}):
{results}

Instructions:
1. Answer the question directly in 1-2 sentences
2. Include specific numbers and key findings
3. No fluff, emojis, or unnecessary explanations
4. Be factual and to the point
5. Only mention patterns if they're significant

Response:""",
                input_variables=["question", "count", "shown", "results"]
            )

            chain = prompt | self.chat_llm | StrOutputParser()
            return chain.invoke({
                "question": question,
                "count": count,
                "shown": len(limited_results),
                "results": json.dumps(limited_results, indent=2, default=str)
            })
        except Exception as e:
            logger.error(f"Detailed response generation failed: {e}")
            return self._generate_basic_response(question, count, results)

    def _generate_summary_response(self, question: str, results: List[Dict], count: int) -> str:
        """Generate summary response for medium result sets"""
        try:
            sample_results = results[:5]

            prompt = PromptTemplate(
                template="""Analyze these MongoDB query results and provide a brief summary.

Question: {question}
Total Results: {count}
Sample Results (first 5):
{sample_results}

Instructions:
1. Answer directly in 2-3 sentences maximum
2. State the total count clearly
3. Mention only the most important finding
4. No explanations about databases or general concepts
5. Be concise and factual

Response:""",
                input_variables=["question", "count", "sample_results"]
            )

            chain = prompt | self.chat_llm | StrOutputParser()
            return chain.invoke({
                "question": question,
                "count": count,
                "sample_results": json.dumps(sample_results, indent=2, default=str)
            })
        except Exception as e:
            logger.error(f"Summary response generation failed: {e}")
            return self._generate_basic_response(question, count, results[:5])

    def _generate_high_level_response(self, question: str, results: List[Dict], count: int) -> str:
        """Generate high-level response for large result sets"""
        try:
            sample_results = results[:3]

            prompt = PromptTemplate(
                template="""Analyze these MongoDB query results and provide a brief summary.

Question: {question}
Total Results: {count}
Sample Results (first 3):
{sample_results}

Instructions:
1. Answer in 1-2 sentences maximum
2. State the total count clearly
3. Be direct and factual
4. No unnecessary explanations

Response:""",
                input_variables=["question", "count", "sample_results"]
            )

            chain = prompt | self.chat_llm | StrOutputParser()
            return chain.invoke({
                "question": question,
                "count": count,
                "sample_results": json.dumps(sample_results, indent=2, default=str)
            })
        except Exception as e:
            logger.error(f"High-level response generation failed: {e}")
            return f"Found {count} results matching your query."

    def _generate_no_results_response(self, question: str) -> str:
        """Generate response when no results found"""
        return f"No documents found matching your query: '{question}'. You may want to check your search criteria or try different keywords."

    def _generate_fallback_response(self, question: str, count: int, results: List[Dict]) -> str:
        """Generate fallback response when other methods fail"""
        if count == 0:
            return f"No results found for: '{question}'"
        elif count == 1:
            return f"Found 1 document matching '{question}'."
        else:
            return f"Found {count} documents matching '{question}'."

    def _generate_basic_response(self, question: str, count: int, results: List[Dict]) -> str:
        """Generate basic response as final fallback"""
        return f"Based on your question '{question}', I found {count} matching documents. " + \
            (f"Here are the results: {json.dumps(results, default=str)}" if count <= 3 else
             "The results contain useful information. Would you like me to help you analyze specific aspects?")

    def _get_response_type(self, count: int) -> str:
        """Determine response type based on result count"""
        if count == 0:
            return "no_results"
        elif count <= 5:
            return "detailed"
        elif count <= 50:
            return "summary"
        else:
            return "high_level"

    def _clean_query_response(self, response_text: str) -> str:
        """Clean and validate the LLM response"""
        response_text = response_text.strip()

        # Remove common wrapper patterns
        response_text = re.sub(r'^```json\s*', '', response_text)
        response_text = re.sub(r'\s*```$', '', response_text)
        response_text = re.sub(r'db\.\w+\.aggregate\(\s*', '', response_text)
        response_text = re.sub(r'\);?\s*$', '', response_text)

        # Fix common issues
        response_text = response_text.replace('None', 'null')
        response_text = response_text.replace("'", '"')

        # Ensure proper array format
        if not response_text.startswith('['):
            response_text = '[' + response_text
        if not response_text.endswith(']'):
            response_text = response_text + ']'

        return response_text

    def _validate_mongodb_query(self, pipeline: List[Dict]) -> bool:
        """Validate MongoDB query for common syntax errors"""
        try:
            for stage in pipeline:
                # Check for common $size operator misuse
                stage_str = json.dumps(stage)
                if '"$size"' in stage_str and '"$gt"' in stage_str:
                    # $size should be used with a number, not with comparison operators
                    logger.warning("Detected potential $size misuse in query")
                    return False

                # Add more validation rules as needed

            return True
        except Exception as e:
            logger.error(f"Query validation error: {e}")
            return False

    def _validate_and_parse_json(self, json_text: str) -> List[Dict]:
        """Validate and parse JSON with error handling"""
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")

            # Try additional cleaning
            cleaned = self._additional_json_cleaning(json_text)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                logger.warning("Using fallback basic query")
                return [{"$limit": 100}]

    def _additional_json_cleaning(self, text: str) -> str:
        """Additional JSON cleaning for edge cases"""
        # Fix unquoted keys
        text = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', text)
        # Fix unquoted string values
        text = re.sub(r':\s*([a-zA-Z_]\w*)\s*([,}])', r': "\1"\2', text)
        # Remove trailing commas
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        return text


# Route functions
def route_after_connection(state: NoSQLState) -> str:
    """Route after MongoDB connection"""
    return END if state.get("error") else "create_optimized_query_prompt"


def route_after_prompt(state: NoSQLState) -> str:
    """Route after query prompt creation"""
    return END if state.get("error") else "intelligent_query_generation"


def route_after_query_gen(state: NoSQLState) -> str:
    """Route after query generation"""
    return END if state.get("error") else "execute_mongodb_query"


def route_after_execution(state: NoSQLState) -> str:
    """Route after query execution - go to judge even if there are query errors"""
    # Always go to judge unless there's a connection error
    if state.get("error") and "connection" in state.get("error", "").lower():
        return END

    # For query execution errors, still generate a response and judge it
    if state.get("error") and not state.get("final_answer"):
        # Generate error response to be judged
        state["final_answer"] = f"Query execution failed: {state['error']}"
        state["result_count"] = 0
        state["response_type"] = "error"

    return "adaptive_conversational_response"


def route_after_response(state: NoSQLState) -> str:
    """Route after response generation"""
    return END if state.get("error") else "response_quality_judge"


def route_after_judge(state: NoSQLState) -> str:
    """Route after quality judgment - decide if we need fallback"""
    if state.get("error"):
        return END

    quality_score = state.get("quality_score", 10)
    fallback_attempt = state.get("fallback_attempt", False)

    # If quality is good or we already tried fallback, end
    if quality_score >= 6 or fallback_attempt:
        return END

    # If quality is poor and we haven't tried fallback, start learning
    logger.info(f"Quality score {quality_score} below threshold, starting fallback learning")
    return "database_learning"


def route_after_learning(state: NoSQLState) -> str:
    """Route after database learning"""
    return END if state.get("error") else "question_rephrase"


def route_after_rephrase(state: NoSQLState) -> str:
    """Route after question rephrasing"""
    if state.get("error"):
        return END

    # Mark as fallback attempt and go back to query generation
    state["fallback_attempt"] = True
    return "intelligent_query_generation"


class NoSQLGraph:
    def __init__(self, azure_api_key: str = None, azure_endpoint: str = None,
                 azure_api_version: str = None, openai_api_key: str = None):

        self.nodes = NoSQLNodes(
            azure_api_key, azure_endpoint, azure_api_version, openai_api_key,
            aws_access_key_id, aws_secret_access_key, aws_region="us-east-1",
            claude_model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        )
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        """Create enhanced LangGraph workflow for NoSQL operations with judge and learning"""
        workflow = StateGraph(NoSQLState)

        # Add all nodes including new ones
        workflow.add_node("initialize_mongodb_connection", self.nodes.initialize_mongodb_connection_node)
        workflow.add_node("create_optimized_query_prompt", self.nodes.create_optimized_query_prompt_node)
        workflow.add_node("intelligent_query_generation", self.nodes.intelligent_query_generation_node)
        workflow.add_node("execute_mongodb_query", self.nodes.execute_mongodb_query_node)
        workflow.add_node("adaptive_conversational_response", self.nodes.adaptive_conversational_response_node)

        # New nodes for enhanced workflow
        workflow.add_node("response_quality_judge", self.nodes.response_quality_judge_node)
        workflow.add_node("database_learning", self.nodes.database_learning_node)
        workflow.add_node("question_rephrase", self.nodes.question_rephrase_node)

        # Set entry point
        workflow.set_entry_point("initialize_mongodb_connection")

        # Main workflow edges
        workflow.add_conditional_edges(
            "initialize_mongodb_connection",
            route_after_connection,
            {
                "create_optimized_query_prompt": "create_optimized_query_prompt",
                END: END
            }
        )

        workflow.add_conditional_edges(
            "create_optimized_query_prompt",
            route_after_prompt,
            {
                "intelligent_query_generation": "intelligent_query_generation",
                END: END
            }
        )

        workflow.add_conditional_edges(
            "intelligent_query_generation",
            route_after_query_gen,
            {
                "execute_mongodb_query": "execute_mongodb_query",
                END: END
            }
        )

        workflow.add_conditional_edges(
            "execute_mongodb_query",
            route_after_execution,
            {
                "adaptive_conversational_response": "adaptive_conversational_response",
                END: END
            }
        )

        workflow.add_conditional_edges(
            "adaptive_conversational_response",
            route_after_response,
            {
                "response_quality_judge": "response_quality_judge",
                END: END
            }
        )

        # Quality judge routing - key decision point
        workflow.add_conditional_edges(
            "response_quality_judge",
            route_after_judge,
            {
                "database_learning": "database_learning",
                END: END
            }
        )

        # Fallback workflow edges
        workflow.add_conditional_edges(
            "database_learning",
            route_after_learning,
            {
                "question_rephrase": "question_rephrase",
                END: END
            }
        )

        workflow.add_conditional_edges(
            "question_rephrase",
            route_after_rephrase,
            {
                "intelligent_query_generation": "intelligent_query_generation",
                END: END
            }
        )

        # Add memory
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def invoke(self, mongo_uri: str, db_name: str, collection_name: str,
               table_schema: str, schema_description: str, few_shot_examples: List[Dict],
               question: str, messages: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Invoke the enhanced NoSQL MongoDB workflow with judge and learning capabilities

        Args:
            mongo_uri: MongoDB connection URI
            db_name: Database name
            collection_name: Collection name
            table_schema: Collection schema description
            schema_description: Additional schema context
            few_shot_examples: List of example queries
            question: User question
            messages: Chat history
        """
        if messages is None:
            messages = []

        initial_state = NoSQLState(
            question=question,
            messages=messages,
            mongo_uri=mongo_uri,
            db_name=db_name,
            collection_name=collection_name,
            table_schema=table_schema,
            schema_description=schema_description,
            few_shot_examples=few_shot_examples,
            collection_stats=None,
            query_prompt_template=None,
            generated_query=None,
            raw_query_response=None,
            query_results=None,
            result_count=0,
            final_answer=None,
            error=None,
            query_context=None,
            execution_stats=None,
            quality_score=None,
            judge_reasoning=None,
            sample_documents=None,
            field_analysis=None,
            learning_examples=None,
            rephrased_question=None,
            fallback_attempt=False,
            exploratory_results=None,
            response_type=None
        )

        # Configure for this conversation
        config = {"configurable": {"thread_id": f"nosql_{hash(question)}"}}

        try:
            result = self.graph.invoke(initial_state, config=config)

            if result.get("error"):
                return {
                    "success": False,
                    "error": result["error"],
                    "question": question
                }

            result = serialize_for_msgpack(result)

            return {
                "success": True,
                "question": question,
                "answer": result["final_answer"],
                "query": result["generated_query"],
                "raw_query": result.get("raw_query_response"),
                "result_count": result.get("result_count", 0),
                "response_type": result.get("response_type", "unknown"),
                "execution_stats": result.get("execution_stats", {}),
                "messages": result["messages"],
                # Enhanced response information
                "quality_score": result.get("quality_score"),
                "used_fallback": result.get("fallback_attempt", False),
                "rephrased_question": result.get("rephrased_question"),
                "learning_applied": bool(result.get("learning_examples")),
                "query_context": result.get("query_context", {})
            }

        except Exception as e:
            logger.error(f"Error in enhanced NoSQL LangGraph execution: {e}")
            return {
                "success": False,
                "error": f"Enhanced NoSQL workflow execution failed: {str(e)}",
                "question": question
            }