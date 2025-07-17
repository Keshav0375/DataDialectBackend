from typing import Dict, List, Any
import json
import re
import os
from bson import ObjectId
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, AzureChatOpenAI, OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma

from pymongo import MongoClient

from app.models.schemas import NoSQLState
from logging_config import setup_logger

logger = setup_logger("DataDialect", "DataDialect.log")

azure_api_key = os.getenv("API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_version = os.getenv("API_VERSION", "2023-05-15")
openai_api_key = os.getenv("OPENAI_API_KEY")


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
    else:
        return obj


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
                 azure_api_version: str = None, openai_api_key: str = None):
        self.azure_api_key = azure_api_key
        self.azure_endpoint = azure_endpoint
        self.azure_api_version = azure_api_version
        self.openai_api_key = openai_api_key

        self.query_llm = self._initialize_query_llm()
        self.chat_llm = self._initialize_chat_llm()
        self.embeddings = self._initialize_embeddings()

    def _initialize_query_llm(self):
        """Initialize LLM for query generation with lower temperature"""
        if self.azure_api_key and self.azure_endpoint and self.azure_api_version:
            return AzureChatOpenAI(
                deployment_name="slideoo-chat-1",
                temperature=0.1,
                max_tokens=4000,
                azure_endpoint=self.azure_endpoint,
                api_key=self.azure_api_key,
                api_version=self.azure_api_version,
            )
        elif self.openai_api_key:
            return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
        else:
            raise ValueError("No valid API key provided for OpenAI or Azure")

    def _initialize_chat_llm(self):
        """Initialize LLM for conversational responses with higher temperature"""
        if self.azure_api_key and self.azure_endpoint and self.azure_api_version:
            return AzureChatOpenAI(
                deployment_name="slideoo-chat-1",
                temperature=0.7,
                max_tokens=2000,
                azure_endpoint=self.azure_endpoint,
                api_key=self.azure_api_key,
                api_version=self.azure_api_version,
            )
        elif self.openai_api_key:
            return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        else:
            raise ValueError("No valid API key provided for OpenAI or Azure")

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

            # Select most relevant examples using embeddings
            relevant_examples = self._select_relevant_examples(question, few_shot_examples)

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
                "user_question": question
            })

            cleaned_response = self._clean_query_response(response)
            pipeline = self._validate_and_parse_json(cleaned_response)

            # Add safety limits
            pipeline = self._add_safety_limits(pipeline)

            state["generated_query"] = pipeline
            state["raw_query_response"] = response
            state["query_context"] = {
                "examples_used": len(relevant_examples),
                "pipeline_stages": len(pipeline)
            }

            logger.info(f"Generated {len(pipeline)} stage pipeline")

        except Exception as e:
            logger.error(f"Error generating MongoDB query: {e}")
            state["error"] = f"Query generation failed: {str(e)}"

        return state

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

    def _add_safety_limits(self, pipeline: List[Dict]) -> List[Dict]:
        """Add safety limits to prevent excessive data retrieval"""
        # Check if pipeline already has limit or group operations
        has_limit = any("$limit" in stage for stage in pipeline)
        has_group = any("$group" in stage for stage in pipeline)
        has_count = any("$count" in stage for stage in pipeline)

        # Add reasonable limit if none exists and it's not an aggregation query
        if not has_limit and not has_group and not has_count:
            pipeline.append({"$limit": 1000})

        return pipeline

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
            results = state["query_results"]
            result_count = state["result_count"]

            # Choose response strategy based on result size
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

    def _generate_detailed_response(self, question: str, results: List[Dict], count: int) -> str:
        """Generate detailed response for small result sets"""
        try:
            prompt = PromptTemplate(
                template="""Analyze these MongoDB query results and provide a concise, direct response.

    Question: {question}
    Results ({count} documents):
    {results}

    Instructions:
    1. Answer the question directly in 1-2 sentences
    2. Include specific numbers and key findings
    3. No fluff, emojis, or unnecessary explanations
    4. Be factual and to the point
    5. Only mention patterns if they're significant

    Response:""",
                input_variables=["question", "count", "results"]
            )

            chain = prompt | self.chat_llm | StrOutputParser()
            return chain.invoke({
                "question": question,
                "count": count,
                "results": json.dumps(results, indent=2, default=str)
            })
        except Exception as e:
            logger.error(f"Detailed response generation failed: {e}")
            return self._generate_basic_response(question, count, results)

    def _generate_summary_response(self, question: str, results: List[Dict], count: int) -> str:
        """Generate summary response for medium result sets"""
        try:
            sample_results = results[:10]

            prompt = PromptTemplate(
                template="""Analyze these MongoDB query results and provide a brief summary.

    Question: {question}
    Total Results: {count}
    Sample Results (first 10):
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
            return self._generate_basic_response(question, count, results[:10])

    def _generate_high_level_response(self, question: str, results: List[Dict], count: int) -> str:
        """Generate high-level response for large result sets"""
        try:
            sample_results = results[:5]

            prompt = PromptTemplate(
                template="""Analyze these MongoDB query results and provide a brief summary.

    Question: {question}
    Total Results: {count}
    Sample Results (first 5):
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
            (f"Here are the results: {json.dumps(results, default=str)}" if count <= 5 else
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
    """Route after query execution"""
    return END if state.get("error") else "adaptive_conversational_response"


class NoSQLGraph:
    def __init__(self, azure_api_key: str = None, azure_endpoint: str = None,
                 azure_api_version: str = None, openai_api_key: str = None):

        self.nodes = NoSQLNodes(
            azure_api_key, azure_endpoint, azure_api_version, openai_api_key
        )
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        """Create optimized LangGraph workflow for NoSQL operations"""
        workflow = StateGraph(NoSQLState)

        # Add nodes
        workflow.add_node("initialize_mongodb_connection", self.nodes.initialize_mongodb_connection_node)
        workflow.add_node("create_optimized_query_prompt", self.nodes.create_optimized_query_prompt_node)
        workflow.add_node("intelligent_query_generation", self.nodes.intelligent_query_generation_node)
        workflow.add_node("execute_mongodb_query", self.nodes.execute_mongodb_query_node)
        workflow.add_node("adaptive_conversational_response", self.nodes.adaptive_conversational_response_node)

        # Set entry point and add conditional edges
        workflow.set_entry_point("initialize_mongodb_connection")

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

        # Final cleanup
        workflow.add_edge("adaptive_conversational_response", END)

        # Add memory
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def invoke(self, mongo_uri: str, db_name: str, collection_name: str,
               table_schema: str, schema_description: str, few_shot_examples: List[Dict],
               question: str, messages: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Invoke the optimized NoSQL MongoDB workflow

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

            return {
                "success": True,
                "question": question,
                "answer": result["final_answer"],
                "query": result["generated_query"],
                "raw_query": result.get("raw_query_response"),
                "result_count": result.get("result_count", 0),
                "response_type": result.get("response_type", "unknown"),
                "execution_stats": result.get("execution_stats", {}),
                "messages": result["messages"]
            }

        except Exception as e:
            logger.error(f"Error in NoSQL LangGraph execution: {e}")
            return {
                "success": False,
                "error": f"NoSQL workflow execution failed: {str(e)}",
                "question": question
            }