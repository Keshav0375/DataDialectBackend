from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict, List, Literal, Optional, Any
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

# ── Pydantic schemas ─────────────────────────────────────────────────
class RouteDecision(BaseModel):
    route: Literal["rag", "answer", "end"]
    reply: str | None = Field(None, description="Filled only when route == 'end'")

class RagJudge(BaseModel):
    sufficient: bool


azure_api_key = os.getenv("API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_version = os.getenv("rag_API_VERSION", "2023-05-15")
openai_api_key = os.getenv("OPENAI_API_KEY")

llm = AzureChatOpenAI(
                deployment_name="slideoo-chat-1",
                temperature=0.1,
                max_tokens=4000,
                azure_endpoint=azure_endpoint,
                api_key=azure_api_key,
                api_version=azure_api_version,
            )

# ── LLM instances with structured output where needed ───────────────

router_llm = llm | StrOutputParser()
judge_llm = llm | StrOutputParser()
answer_llm = AzureChatOpenAI(
                deployment_name="slideoo-chat-1",
                temperature=0.7,
                max_tokens=4000,
                azure_endpoint=azure_endpoint,
                api_key=azure_api_key,
                api_version=azure_api_version,
            )

# ── Shared state type ────────────────────────────────────────────────
class AgentState(TypedDict, total=False):
    """Enhanced agent state supporting document IDs"""
    messages: List[Any]
    route: Optional[str]
    rag: Optional[str]
    web: Optional[str]
    document_ids: Optional[List[int]]
    collection_id: Optional[str]
    context_sources: Optional[List[str]]