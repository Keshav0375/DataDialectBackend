from typing import Dict, List, Any
import ast
from bson import ObjectId
from operator import itemgetter
import os
from langchain_core.runnables import RunnablePassthrough

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools import QuerySQLDatabaseTool  # Fixed import
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, AzureChatOpenAI, OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate, \
    PromptTemplate
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory

from app.database.operations import get_upload_record, update_chat_history
from app.models.schemas import SQLState, TableSchema
from langchain_aws import ChatBedrock
import boto3

from logging_config import setup_logger

logger = setup_logger("DataDialect", "DataDialect.log")


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
    elif isinstance(obj, dict):
        return {k: serialize_for_msgpack(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_msgpack(item) for item in obj]
    else:
        return obj


aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

class SQLNodes:
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

        self.llm = self._initialize_query_llm()
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

    def load_upload_record_node(self, state: SQLState) -> SQLState:
        """Node 1: Load Upload record from MongoDB"""
        logger.info(f"Loading upload record from ID: {state['upload_id']}")
        cache_key = f"upload_record_{state['upload_id']}"

        cached_record = cache_manager.get(cache_key)

        if cached_record:
            logger.info("Using Cache Upload Record")
            state["upload_record"] = cached_record
        else:
            record = get_upload_record(state["upload_id"])
            if not record:
                state["error"] = "Upload ID not found"
                return state

            # Serialize the record to handle ObjectId and other non-serializable objects
            serialized_record = serialize_for_msgpack(record)
            cache_manager.set(cache_key, serialized_record)
            state["upload_record"] = serialized_record

        return state

    def validate_record_status_node(self, state: SQLState) -> SQLState:
        """Node 2: Validate record status"""
        logger.info("Validating record status")

        record = state["upload_record"]
        status = record.get("status")

        if status == "credentials_pending":
            state["error"] = "Database credentials must be uploaded first"
        elif status == "csv_pending":
            state["error"] = "CSV file must be uploaded first"
        elif status == "python_file_pending":
            state["error"] = "Python file must be uploaded first"
        elif status != "completed":
            state["error"] = f"Invalid record status: {status}"

        return state

    def initialize_database_connection_node(self, state: SQLState) -> SQLState:
        """Node 3: Initialize database connection"""
        logger.info("Initializing database connection")

        try:
            record = state["upload_record"]
            credentials = record['credentials']
            db_uri = f"mysql+pymysql://{credentials['db_user']}:{credentials['db_password']}@{credentials['db_host']}/{credentials['db_name']}"
            db = SQLDatabase.from_uri(db_uri)
            db.get_usable_table_names()
            state["database_uri"] = db_uri
            logger.info("Database Connection established successfully")

        except Exception as e:
            logger.error(f"Error initializing database connection: {e}")
            state["error"] = f"Database connection failed: {str(e)}"

        return state

    def get_table_info_node(self, state: SQLState) -> SQLState:
        """Node 4: Get Table Information"""
        logger.info("Getting Table Information")
        cache_key = f"table_info_{state['upload_id']}"
        cached_table_info = cache_manager.get(cache_key)

        if cached_table_info:
            logger.info("Using cached table information")
            state["table_info"] = cached_table_info
            return state

        try:
            record = state["upload_record"]
            csv_data = record.get("csv_data")

            if csv_data:
                table_info = ""
                for table_name, description in [("main_table", "User uploaded data table")]:
                    columns = csv_data.get("columns", [])
                    table_info += f"Table: {table_name}\n"
                    table_info += f"Description: {description}\n"
                    table_info += f"Columns: {', '.join(columns)}\n\n"

                cache_manager.set(cache_key, table_info)
                state["table_info"] = table_info
            else:
                state["error"] = "No CSV data found in record"

        except Exception as e:
            logger.error(f"Error getting table info: {e}")
            state["error"] = f"Failed to get table info: {str(e)}"

        return state

    def table_selection_node(self, state: SQLState) -> SQLState:
        """Node 5: Select Relevant Tables"""
        logger.info("Selecting relevant tables")

        try:
            table_details = ""
            record = state["upload_record"]
            csv_data = record.get("csv_data")

            if not csv_data or "data" not in csv_data:
                raise ValueError("Missing CSV 'data' rows")

            for row in csv_data["data"]:
                table_name = row.get("Table", "Unknown")
                description = row.get("Description", "No description provided")

                table_details += (
                        "Table Name: " + table_name + "\n" +
                        "Table Description: " + description + "\n\n"
                )

            state["relevant_tables"] = table_details
            logger.info(f"Selected tables: {table_details}")

        except Exception as e:
            logger.error(f"Error in table selection: {e}")
            state["error"] = f"Table selection failed: {str(e)}"

        return state

    def table_func(self, state: SQLState):
        """Extract relevant table names from the question"""
        table_details = state["relevant_tables"]

        # Create a simple prompt-based approach instead of using extraction chain
        table_details_prompt = f"""Given the following user question and available tables, return ONLY the names of SQL tables that might be relevant to answering the question.

    Available tables:
    {table_details}

    User question: {state["question"]}

    Instructions:
    - Return only table names, one per line
    - Include ALL tables that might be relevant
    - If unsure, include the table
    - Do not include explanations or additional text

    Table names:"""

        try:
            # Use direct LLM call instead of extraction chain for Bedrock compatibility
            from langchain_core.prompts import PromptTemplate
            from langchain_core.output_parsers import StrOutputParser

            prompt = PromptTemplate.from_template(table_details_prompt)
            chain = prompt | self.llm | StrOutputParser()

            response = chain.invoke({"question": state["question"]})

            # Parse the response to extract table names
            table_names = [name.strip() for name in response.split('\n') if name.strip()]

            # Create a mock TableSchema response to maintain compatibility
            # Assuming TableSchema has a 'table_names' field
            class MockTableSchema:
                def __init__(self, table_names):
                    self.table_names = table_names

            return MockTableSchema(table_names)

        except Exception as e:
            logger.error(f"Error in table_func: {e}")
            # Fallback: return all available tables
            all_tables = []
            for line in table_details.split('\n'):
                if line.startswith('Table Name:'):
                    table_name = line.replace('Table Name:', '').strip()
                    all_tables.append(table_name)

            class MockTableSchema:
                def __init__(self, table_names):
                    self.table_names = table_names

            return MockTableSchema(all_tables)


    def get_example_selector(self, state: SQLState):
        """Get Example Selector for few shot prompting"""
        try:
            record = state["upload_record"]
            python_file = record.get("python_file", {})
            content = python_file.get("content", "")

            if not content:
                raise ValueError("No Python content found in upload record")

            parsed = ast.parse(content)

            examples = []
            for node in ast.walk(parsed):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "examples":
                            examples = ast.literal_eval(node.value)

            if not isinstance(examples, list):
                raise ValueError("No valid 'examples' list found in python_file content")

            state["relevant_examples"] = examples
            logger.info(f"Extracted {len(examples)} example(s) from python_file")

            if self.embeddings:
                example_selector = SemanticSimilarityExampleSelector.from_examples(
                    examples,
                    self.embeddings,
                    Chroma,
                    k=2,
                    input_keys=["input"],
                )
                return example_selector

        except Exception as e:
            logger.error(f"Failed to extract examples: {e}")
            state["error"] = f"Example selector failed: {str(e)}"

        return None

    def query_generation_node(self, state: SQLState) -> SQLState:
        """Node 6: Generate SQL Query"""
        logger.info("Generating SQL Query")
        try:
            question = state["question"]
            record = state["upload_record"]
            credentials = record['credentials']
            db_uri = f"mysql+pymysql://{credentials['db_user']}:{credentials['db_password']}@{credentials['db_host']}/{credentials['db_name']}"
            db = SQLDatabase.from_uri(db_uri)

            example_prompt = ChatPromptTemplate.from_messages(
                [
                    ("human", "{input}\nSQLQuery:"),
                    ("ai", "{query}"),
                ]
            )
            example_selector = self.get_example_selector(state)

            if example_selector:
                few_shot_prompt = FewShotChatMessagePromptTemplate(
                    example_prompt=example_prompt,
                    example_selector=example_selector,
                    input_variables=["input", "top_k"],
                )
            else:
                few_shot_prompt = None

            if few_shot_prompt:
                final_prompt = ChatPromptTemplate.from_messages([
                    ("system",
                     "You are a MySQL expert. Given an input question, create a syntactically correct MySQL query to run."
                     " Unless otherwise specified.\n\nHere is the relevant table info: {table_info}\n\n"
                     "Below are a number of examples of questions and their corresponding SQL queries."),
                    few_shot_prompt,
                    MessagesPlaceholder(variable_name="messages"),
                    ("human", "{input}"),
                ])
            else:
                final_prompt = ChatPromptTemplate.from_messages([
                    ("system",
                     "You are a MySQL expert. Given an input question, create a syntactically correct MySQL query to run."
                     " Unless otherwise specified.\n\nHere is the relevant table info: {table_info}"),
                    MessagesPlaceholder(variable_name="messages"),
                    ("human", "{input}"),
                ])

            generate_query = create_sql_query_chain(self.llm, db, final_prompt)

            history = ChatMessageHistory()

            for msg in state.get("messages", []):
                if msg["role"] == "user":
                    history.add_user_message(msg["content"])
                else:
                    history.add_ai_message(msg["content"])

            query = generate_query.invoke({
                "question": question,
                "top_k": 3,
                "messages": history.messages,
                "table_info": state["relevant_tables"]
            })

            state["generated_query"] = query
            logger.info(f"Generated query: {query}")

        except Exception as e:
            logger.error(f"Error generating query: {e}")
            state["error"] = f"Query generation failed: {str(e)}"

        return state

    def query_execution_node(self, state: SQLState) -> SQLState:
        """Node 7: Execute SQL Query"""
        logger.info("Executing SQL Query")

        try:
            query = state["generated_query"]
            if not query:
                raise ValueError("No query generated")

            record = state["upload_record"]
            credentials = record['credentials']
            db_uri = f"mysql+pymysql://{credentials['db_user']}:{credentials['db_password']}@{credentials['db_host']}/{credentials['db_name']}"
            db = SQLDatabase.from_uri(db_uri)
            execute_query = QuerySQLDatabaseTool(db=db)

            # Extract just the SQL query if there's additional text
            sql_query = query
            if "SELECT" in query.upper():
                # Find the actual SQL query in the response
                lines = query.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.upper().startswith('SELECT'):
                        sql_query = line
                        # Check if query continues on next lines
                        line_idx = lines.index(line)
                        for next_line in lines[line_idx + 1:]:
                            next_line = next_line.strip()
                            if next_line and not next_line.startswith('#') and not next_line.startswith('--'):
                                if any(keyword in next_line.upper() for keyword in
                                       ['FROM', 'WHERE', 'JOIN', 'ORDER BY', 'GROUP BY', 'HAVING', 'LIMIT']):
                                    sql_query += ' ' + next_line
                                else:
                                    break
                        break

            # Execute the query directly
            try:
                result = execute_query.run(sql_query)
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                # Try to clean up the query and execute again
                cleaned_query = sql_query.replace('\n', ' ').strip()
                if cleaned_query.endswith(';'):
                    cleaned_query = cleaned_query[:-1]
                result = execute_query.run(cleaned_query)

            # Create the answer prompt
            answer_prompt = PromptTemplate.from_template(
                """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

                Question: {question}
                SQL Query: {query}
                SQL Result: {result}
                Answer: """
            )

            rephrase_answer = answer_prompt | self.llm | StrOutputParser()

            # Generate the final answer
            final_result = rephrase_answer.invoke({
                "question": state["question"],
                "query": sql_query,
                "result": result
            })

            state["query_result"] = final_result
            logger.info("Query executed Successfully")

        except Exception as e:
            logger.error(f"Error executing query: {e}")
            state["error"] = f"Query execution failed: {str(e)}"

        return state

    def answer_generation_node(self, state: SQLState) -> SQLState:
        """Node 8: Generate final answer (now optional since query_execution might handle this)"""
        logger.info("Generating Final Answer")

        try:
            state["final_answer"] = state["query_result"]
            logger.info("Final answer generated successfully")
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            state["error"] = f"Answer generation failed: {str(e)}"

        return state

    def update_memory_node(self, state: SQLState) -> SQLState:
        """Node 9: Update memory/chat history"""
        logger.info("Updating memory")

        try:
            messages = state.get("messages", [])
            messages.append({"role": "user", "content": state["question"]})
            messages.append({"role": "assistant", "content": state["final_answer"]})

            # Try different parameter combinations based on the function signature
            try:
                # Option 1: Three parameters (upload_id, question, messages)
                update_chat_history(state["upload_id"], state["question"], messages)
            except TypeError:
                try:
                    # Option 2: Two parameters (upload_id, messages)
                    update_chat_history(state["upload_id"], messages)
                except TypeError:
                    # Option 3: Different parameter order or names
                    # Log the error and continue without updating chat history
                    logger.warning("Could not determine correct parameters for update_chat_history. Skipping chat history update.")

            state["messages"] = messages
            logger.info("Memory updated successfully")

        except Exception as e:
            logger.error(f"Error updating memory: {e}")
            state["error"] = f"Memory update failed: {str(e)}"

        return state




def route_after_validation(state: SQLState) -> str:
    """Route after validation based on error state"""
    if state.get("error"):
        return END
    return "initialize_database_connection"


def route_after_execution(state: SQLState) -> str:
    """Route after query execution"""
    if state.get("error"):
        return END
    return "answer_generation"


class SQLGraph:
    def __init__(self, azure_api_key: str = None, azure_endpoint: str = None,
                 azure_api_version: str = None, openai_api_key: str = None):

        self.nodes = SQLNodes(
            azure_api_key, azure_endpoint, azure_api_version, openai_api_key,
            aws_access_key_id, aws_secret_access_key, aws_region="us-east-1",
            claude_model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        )

        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(SQLState)

        workflow.add_node("load_upload_record", self.nodes.load_upload_record_node)
        workflow.add_node("validate_record_status", self.nodes.validate_record_status_node)
        workflow.add_node("initialize_database_connection", self.nodes.initialize_database_connection_node)
        workflow.add_node("get_table_info", self.nodes.get_table_info_node)
        workflow.add_node("table_selection", self.nodes.table_selection_node)
        workflow.add_node("query_generation", self.nodes.query_generation_node)
        workflow.add_node("query_execution", self.nodes.query_execution_node)
        workflow.add_node("answer_generation", self.nodes.answer_generation_node)
        workflow.add_node("update_memory", self.nodes.update_memory_node)

        # Add edges
        workflow.set_entry_point("load_upload_record")
        workflow.add_edge("load_upload_record", "validate_record_status")
        workflow.add_conditional_edges(
            "validate_record_status",
            route_after_validation,
            {
                "initialize_database_connection": "initialize_database_connection",
                END: END
            }
        )
        workflow.add_edge("initialize_database_connection", "get_table_info")
        workflow.add_edge("get_table_info", "table_selection")
        workflow.add_edge("table_selection", "query_generation")
        workflow.add_edge("query_generation", "query_execution")
        workflow.add_conditional_edges(
            "query_execution",
            route_after_execution,
            {
                "answer_generation": "answer_generation",
                END: END
            }
        )
        workflow.add_edge("answer_generation", "update_memory")
        workflow.add_edge("update_memory", END)

        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def invoke(self, upload_id: str, question: str, messages: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Invoke the DataDialect workflow"""
        if messages is None:
            messages = []

        initial_state = SQLState(
            upload_id=upload_id,
            question=question,
            messages=messages,
            upload_record=None,
            database_connection=None,
            database_uri=None,  # Added this field
            table_info=None,
            relevant_tables="",
            generated_query=None,
            query_result=None,
            final_answer=None,
            error=None,
            cached_data={}
        )

        # Configure for this conversation
        config = {"configurable": {"thread_id": upload_id}}

        try:
            result = self.graph.invoke(initial_state, config=config)

            if result.get("error"):
                return {
                    "success": False,
                    "error": result["error"],
                    "upload_id": upload_id
                }

            return {
                "success": True,
                "answer": result["final_answer"],
                "query": result["generated_query"],
                "upload_id": upload_id,
                "messages": result["messages"]
            }

        except Exception as e:
            logger.error(f"Error in LangGraph execution: {e}")
            return {
                "success": False,
                "error": f"Workflow execution failed: {str(e)}",
                "upload_id": upload_id
            }

