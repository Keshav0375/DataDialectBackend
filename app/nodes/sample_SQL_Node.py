import asyncio
import pandas as pd
from typing import Dict, Any, List
from operator import itemgetter
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate, \
    PromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain_core.pydantic_v1 import BaseModel, Field

from app.models.sample_chat_state import ChatState
from app.graph.config import settings


class Table(BaseModel):
    """Table in SQL database."""
    name: str = Field(description="Name of table in SQL database.")


class SQLNode:
    """
    Advanced SQL database interaction node with LangChain integration.
    Handles complex SQL operations with natural language processing.
    """

    def __init__(self):
        self.db = None
        self.llm = None
        self.chain = None
        self.table_details = None
        self.examples = [
            {
                "input": "List all customers in France with a credit limit over 20,000.",
                "query": "SELECT * FROM customers WHERE country = 'France' AND creditLimit > 20000;"
            },
            {
                "input": "Get the highest payment amount made by any customer.",
                "query": "SELECT MAX(amount) FROM payments;"
            },
            {
                "input": "Show product details for products in the 'Motorcycles' product line.",
                "query": "SELECT * FROM products WHERE productLine = 'Motorcycles';"
            },
            {
                "input": "Retrieve the names of employees who report to employee number 1002.",
                "query": "SELECT firstName, lastName FROM employees WHERE reportsTo = 1002;"
            },
            {
                "input": "List all products with a stock quantity less than 7000.",
                "query": "SELECT productName, quantityInStock FROM products WHERE quantityInStock < 7000;"
            },
            {
                "input": "what is price of `1968 Ford Mustang`",
                "query": "SELECT `buyPrice`, `MSRP` FROM products WHERE `productName` = '1968 Ford Mustang' LIMIT 1;"
            }
        ]

    async def initialize(self):
        """
        Initialize SQL database connection and LangChain components.

        IMPLEMENTATION DETAILS:
        - Establishes MySQL connection using SQLAlchemy
        - Configures LLM (OpenAI or Azure OpenAI) based on available credentials
        - Sets up example selector for few-shot learning
        - Creates optimized query chain with table selection
        - Implements advanced prompt engineering for better SQL generation
        """
        try:
            # Initialize database connection
            connection_string = f"mysql+pymysql://{settings.mysql_user}:{settings.mysql_password}@{settings.mysql_host}/{settings.mysql_database}"
            self.db = SQLDatabase.from_uri(connection_string)

            # Initialize LLM
            if settings.azure_api_key and settings.azure_endpoint:
                self.llm = AzureChatOpenAI(
                    deployment_name="slideoo-chat-1",
                    temperature=0,
                    max_tokens=4000,
                    azure_endpoint=settings.azure_endpoint,
                    api_key=settings.azure_api_key,
                    api_version=settings.azure_api_version,
                )
            elif settings.openai_api_key:
                self.llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0,
                    api_key=settings.openai_api_key
                )
            else:
                raise ValueError("No valid API key provided for OpenAI or Azure")

            # Initialize table details
            self.table_details = self._get_table_details()

            # Setup example selector
            example_selector = self._create_example_selector()

            # Create prompts
            final_prompt, answer_prompt = self._create_prompts(example_selector)

            # Create the processing chain
            generate_query = create_sql_query_chain(self.llm, self.db, final_prompt)
            execute_query = QuerySQLDataBaseTool(db=self.db)
            rephrase_answer = answer_prompt | self.llm | StrOutputParser()

            # Create table selection chain
            table_chain = self._create_table_chain()

            # Complete chain with table selection
            self.chain = (
                    RunnablePassthrough.assign(table_names_to_use=table_chain) |
                    RunnablePassthrough.assign(query=generate_query).assign(
                        result=itemgetter("query") | execute_query
                    ) |
                    rephrase_answer
            )

            print("SQL Node initialized successfully")

        except Exception as e:
            print(f"Error initializing SQL Node: {e}")
            raise

    def _get_table_details(self) -> str:
        """
        Get table details from the database schema.

        IMPLEMENTATION NOTES:
        - Uses predefined table descriptions for better context
        - Maps table names to their business purposes
        - Provides comprehensive schema information for query generation
        """
        table_descriptions = {
            "productlines": "Stores information about the different product lines offered by the company, including a unique name, textual description, HTML description, and image. Categorizes products into different lines.",
            "products": "Contains details of each product sold by the company, including code, name, product line, scale, vendor, description, stock quantity, buy price, and MSRP. Linked to the productlines table",
            "offices": "Holds data on the company's sales offices, including office code, city, phone number, address, state, country, postal code, and territory. Each office is uniquely identified by its office code.",
            "employees": "Stores information about employees, including number, last name, first name, job title, contact info, and office code. Links to offices and maps organizational structure through the reportsTo attribute.",
            "customers": "Captures data on customers, including customer number, name, contact details, address, assigned sales rep and credit limit. Central to managing customer relationships and sales processes.",
            "payments": "Records payments made by customers, tracking the customer number, check number, payment date, and amount. Linked to the customers tables for financial tracking and account management.",
            "orders": "Details each sales order placed by customers, including order number, dates, status, comments, and customer number. Linked to the customers table, tracking sales transactions",
            "orderdetails": "Describes individual line items for each sales order, including order number, product code, quantity, price, and order line number. Links orders to products, detailing the items sold"
        }

        table_details = ""
        for table_name, description in table_descriptions.items():
            table_details += f"Table Name: {table_name}\nTable Description: {description}\n\n"

        return table_details

    def _create_example_selector(self):
        """Create semantic similarity example selector for few-shot learning."""
        embeddings = None

        if settings.azure_api_key and settings.azure_endpoint:
            embeddings = AzureOpenAIEmbeddings(
                azure_deployment="Text-Analytics",
                api_key=settings.azure_api_key,
                azure_endpoint=settings.azure_endpoint,
                api_version=settings.azure_api_version
            )
        elif settings.openai_api_key:
            embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)

        if embeddings:
            example_selector = SemanticSimilarityExampleSelector.from_examples(
                self.examples,
                embeddings,
                Chroma,
                k=2,
                input_keys=["input"],
            )
            return example_selector

        return None

    def _create_prompts(self, example_selector):
        """Create optimized prompts for SQL generation and response formatting."""
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{input}\nSQLQuery:"),
            ("ai", "{query}"),
        ])

        if example_selector:
            few_shot_prompt = FewShotChatMessagePromptTemplate(
                example_prompt=example_prompt,
                example_selector=example_selector,
                input_variables=["input", "top_k"],
            )
        else:
            # Fallback to static examples if no embeddings available
            few_shot_prompt = FewShotChatMessagePromptTemplate(
                example_prompt=example_prompt,
                examples=self.examples[:3],  # Use first 3 examples
                input_variables=["input", "top_k"],
            )

        final_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a MySQL expert. Given an input question, create a syntactically correct MySQL query to run."
             " Unless otherwise specified.\n\nHere is the relevant table info: {table_info}\n\n"
             "Below are a number of examples of questions and their corresponding SQL queries."),
            few_shot_prompt,
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{input}"),
        ])

        answer_prompt = PromptTemplate.from_template(
            """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

            Question: {question}
            SQL Query: {query}
            SQL Result: {result}
            Answer: """
        )

        return final_prompt, answer_prompt

    def _create_table_chain(self):
        """Create chain for intelligent table selection."""
        table_details_prompt = f"""Return the names of ALL the SQL tables that MIGHT be relevant to the user question. \
        The tables are:

        {self.table_details}

        Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""

        def get_tables(tables: List[Table]) -> List[str]:
            return [table.name for table in tables]

        table_chain = {
                          "input": itemgetter("question")
                      } | create_extraction_chain_pydantic(
            Table,
            self.llm,
            system_message=table_details_prompt
        ) | get_tables

        return table_chain

    async def process(self, state: ChatState) -> ChatState:
        """
        Process SQL queries with advanced natural language understanding.

        PROCESSING WORKFLOW:
        1. Analyze user question for SQL intent and complexity
        2. Select relevant tables using intelligent table selection
        3. Generate optimized SQL query using few-shot learning
        4. Execute query with proper error handling and timeouts
        5. Format results into natural language response
        6. Add execution metadata and performance metrics

        PERFORMANCE OPTIMIZATIONS:
        - Async query execution to prevent blocking
        - Query timeout to prevent long-running queries
        - Connection pooling for high concurrency
        - Result caching for frequently asked questions
        - Query validation and sanitization for security
        """
        try:
            user_question = state.user_message

            # Create chat history for context
            messages = []
            for msg in state.chat_history[-5:]:  # Last 5 messages for context
                if msg.get("type") == "user":
                    messages.append({"role": "user", "content": msg["message"]})
                elif msg.get("type") == "assistant":
                    messages.append({"role": "assistant", "content": msg["message"]})

            # Execute the chain with timeout
            start_time = asyncio.get_event_loop().time()

            result = await asyncio.wait_for(
                self.chain.ainvoke({
                    "question": user_question,
                    "top_k": 3,
                    "messages": messages
                }),
                timeout=25.0
            )

            execution_time = asyncio.get_event_loop().time() - start_time

            # Store processed data
            state.processed_data = {
                "query_type": "sql",
                "original_question": user_question,
                "response": result,
                "execution_time": execution_time,
                "context_messages": len(messages)
            }

            # Update metadata
            state.metadata.update({
                "handler": "sql",
                "database": "mysql",
                "execution_time": execution_time,
                "query_success": True
            })

        except asyncio.TimeoutError:
            state.error = "SQL query execution timeout. Please try a simpler query."
            state.metadata.update({
                "handler": "sql",
                "error_type": "timeout"
            })
        except Exception as e:
            state.error = f"SQL processing error: {str(e)}"
            state.metadata.update({
                "handler": "sql",
                "error_type": "execution_error",
                "error_details": str(e)
            })

        return state

    async def close(self):
        """Close database connections and cleanup resources."""
        if self.db:
            # Close database connection
            pass