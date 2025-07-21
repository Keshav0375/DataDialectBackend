from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import BaseMessage
from typing import List
import os
from dotenv import load_dotenv
load_dotenv()

azure_api_key = os.getenv("API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_version = os.getenv("API_VERSION", "2023-05-15")
openai_api_key = os.getenv("OPENAI_API_KEY")

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is. "
    "⟹ Return **only** the reformulated question (no explanations, no answers)."
)

CONTEXT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

def contextualize_q_chain(input_dict):
    """Wrapper to handle empty chat history"""
    chat_history = input_dict.get("chat_history", [])
    if not chat_history:
        return input_dict["input"]
    return contextualise_chain.invoke(input_dict)

contextualise_chain = ( CONTEXT_PROMPT | AzureChatOpenAI(
                deployment_name="slideoo-chat-1",
                temperature=0.1,
                max_tokens=4000,
                azure_endpoint=azure_endpoint,
                api_key=azure_api_key,
                api_version=azure_api_version,
            ) | StrOutputParser()).with_config(run_name="contextualise_chain")