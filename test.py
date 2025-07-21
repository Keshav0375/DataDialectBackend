from langchain_openai import AzureChatOpenAI
import pymongo
from pymongo import MongoClient
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import re
import json
import logging
from datetime import datetime
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigData:
    MONGO_DB_URI = "mongodb://localhost:27017/"
    DB_NAME = "DataDialect"
    COLLECTION_NAME = "NL2SQL"
    TABLE_SCHEMA = '''
        "credentials": {
            "db_host": "string",
            "db_user": "string",
            "db_password": "string",
            "db_name": "string"
        },
        "csv_data": {
            "filename": "string",
            "columns": ["string"],
            "row_count": "integer",
            "data": [
                {
                    "Table": "string",
                    "Description": "string"
                }
            ],
            "full_data_size": "integer"
        },
        "python_file": {
            "filename": "string",
            "content": "string",
            "file_size": "integer",
            "line_count": "integer"
        },
        "created_at": "ISODate string",
        "updated_at": "ISODate string",
        "status": "string"
    '''

    SCHEMA_DESCRIPTION = '''
        Schema Details:

        1. credentials: Database connection information (host, user, password, database name)
        2. csv_data: Information about uploaded CSV files containing table schemas
        3. python_file: Python code examples and scripts
        4. created_at/updated_at: Timestamps for record tracking
        5. status: Processing status of the record
    '''

    FEW_SHOT_EXAMPLES = [
        {
            "input": "Find all completed database schemas that have more than 5 tables",
            "query": [
                {
                    "$match": {
                        "status": "completed",
                        "csv_data.row_count": {"$gt": 5}
                    }
                },
                {
                    "$project": {
                        "credentials.db_name": 1,
                        "csv_data.filename": 1,
                        "csv_data.row_count": 1,
                        "status": 1
                    }
                }
            ]
        },
        {
            "input": "Count how many documents have csv_data and how many have null csv_data",
            "query": [
                {
                    "$group": {
                        "_id": None,
                        "total_with_csv_data": {
                            "$sum": {
                                "$cond": [
                                    {"$and": [
                                        {"$ne": ["$csv_data", None]},
                                        {"$gt": ["$csv_data.row_count", 0]}
                                    ]},
                                    1,
                                    0
                                ]
                            }
                        },
                        "total_with_null_csv_data": {
                            "$sum": {
                                "$cond": [
                                    {"$eq": ["$csv_data", None]},
                                    1,
                                    0
                                ]
                            }
                        }
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "total_with_csv_data": 1,
                        "total_with_null_csv_data": 1
                    }
                }
            ]
        }
    ]


class ConversationalDataChatbot:
    def __init__(self):
        load_dotenv()

        # Initialize Azure OpenAI for query generation
        self.query_llm = AzureChatOpenAI(
            deployment_name="slideoo-chat-1",
            temperature=0.1,
            max_tokens=4000,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("API_KEY"),
            api_version=os.getenv("API_VERSION"),
        )

        # Initialize Azure OpenAI for conversational responses
        self.chat_llm = AzureChatOpenAI(
            deployment_name="slideoo-chat-1",
            temperature=0.7,  # Higher temperature for more natural conversation
            max_tokens=2000,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("API_KEY"),
            api_version=os.getenv("API_VERSION"),
        )

        # Query generation prompt
        self.query_prompt = PromptTemplate(
            template="""
You are an expert MongoDB query specialist. Create MongoDB aggregation pipelines based on user questions.

IMPORTANT RULES:
1. Return ONLY a valid JSON array containing the MongoDB pipeline stages
2. Use null instead of None for null values
3. Do not include any wrapper text, explanations, or markdown
4. Ensure all field names use underscores, not asterisks or other characters

Table Schema: {table_schema}
Schema Description: {schema_description}
Examples: {examples}

User Question: {user_question}

Return only the MongoDB aggregation pipeline as a JSON array:
            """,
            input_variables=["table_schema", "schema_description", "examples", "user_question"],
        )

        # Conversational response prompt
        self.chat_prompt = PromptTemplate(
            template="""
You are a friendly and knowledgeable data assistant helping users understand their database information. 
Your job is to interpret MongoDB query results and present them in a conversational, easy-to-understand way.

Context about the database:
- This is a DataDialect database that stores information about database schemas, CSV files, and Python code examples
- Users can upload CSV files containing table descriptions and generate Python code examples
- The data includes database credentials, CSV metadata, Python files, and processing status

User's Original Question: {user_question}

MongoDB Query Results: {query_results}

Instructions:
1. Analyze the query results and provide a clear, conversational explanation
2. Use natural language and avoid technical jargon when possible
3. If there are numbers/counts, present them clearly with context
4. If there are multiple results, summarize the key findings
5. Be helpful and suggest follow-up questions if relevant
6. If no results were found, explain what that means in a friendly way
7. Use emojis sparingly but appropriately to make responses more engaging

Respond as a helpful data assistant would:
            """,
            input_variables=["user_question", "query_results"],
        )

        # Create chains
        self.query_chain = self.query_prompt | self.query_llm | StrOutputParser()
        self.chat_chain = self.chat_prompt | self.chat_llm | StrOutputParser()

        # Initialize MongoDB connection
        self.client = pymongo.MongoClient(ConfigData.MONGO_DB_URI)
        self.db = self.client[ConfigData.DB_NAME]
        self.collection = self.db[ConfigData.COLLECTION_NAME]

    def clean_response(self, response_text):
        """Clean and validate the LLM response"""
        response_text = response_text.strip()

        # Remove common prefixes/suffixes
        response_text = re.sub(r'^Output:\s*', '', response_text)
        response_text = re.sub(r'^```json\s*', '', response_text)
        response_text = re.sub(r'\s*```$', '', response_text)

        # Remove MongoDB wrapper if present
        response_text = re.sub(r'db\.\w+\.aggregate\(\s*', '', response_text)
        response_text = re.sub(r'\);?\s*$', '', response_text)

        # Fix common formatting issues
        response_text = response_text.replace('*', '_')
        response_text = response_text.replace('None', 'null')

        # Ensure proper array format
        response_text = response_text.strip()
        if not response_text.startswith('['):
            response_text = '[' + response_text
        if not response_text.endswith(']'):
            response_text = response_text + ']'

        return response_text

    def validate_and_parse_json(self, json_text):
        """Validate and parse JSON with error handling"""
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            # Try additional cleaning
            cleaned_text = self.additional_json_cleaning(json_text)
            try:
                return json.loads(cleaned_text)
            except json.JSONDecodeError as e2:
                raise ValueError(f"Unable to parse LLM response as valid JSON: {e2}")

    def additional_json_cleaning(self, text):
        """Additional JSON cleaning for problematic cases"""
        text = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', text)
        text = re.sub(r':\s*([a-zA-Z_]\w*)\s*([,}])', r': "\1"\2', text)
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        return text

    def generate_query(self, user_question):
        """Generate MongoDB query from natural language question"""
        try:
            response = self.query_chain.invoke({
                "table_schema": ConfigData.TABLE_SCHEMA,
                "schema_description": ConfigData.SCHEMA_DESCRIPTION,
                "examples": json.dumps(ConfigData.FEW_SHOT_EXAMPLES, indent=2),
                "user_question": user_question
            })

            cleaned_response = self.clean_response(response)
            query_pipeline = self.validate_and_parse_json(cleaned_response)

            return query_pipeline

        except Exception as e:
            logger.error(f"Error generating query: {str(e)}")
            raise

    def execute_query(self, pipeline):
        """Execute MongoDB aggregation pipeline"""
        try:
            results = list(self.collection.aggregate(pipeline))
            return results
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise

    def format_results_for_chat(self, results):
        """Format query results for better readability in chat"""
        if not results:
            return "No results found"

        # Convert ObjectId and datetime objects to strings for better readability
        formatted_results = []
        for result in results:
            formatted_result = {}
            for key, value in result.items():
                if hasattr(value, '__dict__'):  # ObjectId, datetime, etc.
                    formatted_result[key] = str(value)
                else:
                    formatted_result[key] = value
            formatted_results.append(formatted_result)

        return json.dumps(formatted_results, indent=2, default=str)

    def generate_conversational_response(self, user_question, query_results):
        """Generate a conversational response explaining the query results"""
        try:
            formatted_results = self.format_results_for_chat(query_results)

            response = self.chat_chain.invoke({
                "user_question": user_question,
                "query_results": formatted_results
            })

            return response.strip()

        except Exception as e:
            logger.error(f"Error generating conversational response: {str(e)}")
            return f"I found some results for your question, but I'm having trouble explaining them right now. Here's the raw data: {query_results}"

    def chat(self, user_question):
        """Main chat interface - processes question and returns conversational response"""
        try:
            print(f"\n🤔 Let me search through your database for: '{user_question}'")

            # Generate and execute query
            pipeline = self.generate_query(user_question)
            print(f"✅ Generated query pipeline")

            results = self.execute_query(pipeline)
            print(f"📊 Found {len(results)} result(s)")

            # Generate conversational response
            conversational_response = self.generate_conversational_response(user_question, results)

            return conversational_response

        except Exception as e:
            error_message = f"I'm sorry, I encountered an issue while processing your question: {str(e)}"
            logger.error(error_message)
            return error_message

    def start_interactive_chat(self):
        """Start an interactive chat session"""
        print("🤖 Hello! I'm your DataDialect assistant. I can help you explore your database in a conversational way.")
        print("💡 Try asking questions like:")
        print("   - 'How many database schemas do we have?'")
        print("   - 'Show me completed projects'")
        print("   - 'Which databases have the most tables?'")
        print("   - 'What Python files do we have?'")
        print("\nType 'quit' to exit.\n")

        while True:
            try:
                user_input = input("🗣️ You: ").strip()

                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye! Have a great day!")
                    break

                if not user_input:
                    print("💭 Please ask me a question about your database!")
                    continue

                # Get conversational response
                response = self.chat(user_input)
                print(f"\n🤖 Assistant: {response}\n")
                print("-" * 60)

            except KeyboardInterrupt:
                print("\n👋 Goodbye! Have a great day!")
                break
            except Exception as e:
                print(f"❌ Oops! Something went wrong: {str(e)}")

    def close_connection(self):
        """Close MongoDB connection"""
        if hasattr(self, 'client'):
            self.client.close()


# Example usage functions
def quick_question_example():
    """Example of asking a single question"""
    chatbot = ConversationalDataChatbot()

    try:
        question = "How many objects have csv data and how many have NULL?"
        print(f"Question: {question}")
        response = chatbot.chat(question)
        print(f"\nResponse: {response}")

    finally:
        chatbot.close_connection()


def multiple_questions_example():
    """Example of asking multiple questions"""
    chatbot = ConversationalDataChatbot()

    questions = [
        "How many database schemas are stored in total?",
        "Show me all completed projects",
        "Which databases have more than 10 tables?",
        "What Python files do we have with JOIN operations?"
    ]

    try:
        for question in questions:
            print(f"\n{'=' * 60}")
            print(f"Q: {question}")
            response = chatbot.chat(question)
            print(f"A: {response}")

    finally:
        chatbot.close_connection()


# Main execution
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        # Start interactive chat
        chatbot = ConversationalDataChatbot()
        try:
            chatbot.start_interactive_chat()
        finally:
            chatbot.close_connection()
    elif len(sys.argv) > 1 and sys.argv[1] == "multiple":
        # Run multiple questions example
        multiple_questions_example()
    else:
        # Run single question example
        quick_question_example()