from langgraph.graph import StateGraph, END
from typing import Literal, List, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.utils.shared import AgentState, router_llm, judge_llm, answer_llm
from app.tools.web_search_tool import web_search_tool
from app.tools.rag_tool import rag_search_tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
import os, json
from logging_config import setup_logger

load_dotenv()

logger = setup_logger("DataDialect", "DataDialect.log")

azure_api_key = os.getenv("API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_version = os.getenv("rag_API_VERSION", "2023-05-15")
openai_api_key = os.getenv("OPENAI_API_KEY")


class RAGNodes:
    def __init__(self, azure_api_key: str = None, azure_endpoint: str = None,
                 azure_api_version: str = None, openai_api_key: str = None):
        self.azure_api_key = azure_api_key
        self.azure_endpoint = azure_endpoint
        self.azure_api_version = azure_api_version
        self.openai_api_key = openai_api_key

        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        """Initialize LLM based on available credentials"""
        if self.azure_api_key and self.azure_endpoint and self.azure_api_version:
            return AzureChatOpenAI(
                deployment_name="slideoo-chat-1",
                temperature=1,
                max_tokens=4000,
                azure_endpoint=self.azure_endpoint,
                api_key=self.azure_api_key,
                api_version=self.azure_api_version,
            )
        elif self.openai_api_key:
            return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        else:
            raise ValueError("No valid API key provided for OpenAI or Azure")

    def router_node(self, state: AgentState) -> AgentState:
        """Enhanced router node with document awareness"""
        document_context = ""
        if state.get("document_ids"):
            document_context = f"\nNote: User has uploaded {len(state['document_ids'])} document(s) for analysis."

        system_prompt = (
            "You are a router that decides how to handle user queries. "
            f"{document_context}"
            "Respond with ONLY a valid JSON object in this exact format:\n"
            '{"route": "rag|answer|end", "reply": "your_reply_if_route_is_end"}\n\n'
            "Rules:\n"
            "- Use 'end' for pure greetings/small-talk (include a 'reply')\n"
            "- Use 'rag' when knowledge base lookup is needed OR when user has uploaded documents\n"
            "- Use 'answer' when you can answer directly without external info AND no documents are uploaded\n"
            "- Set 'reply' to null unless route is 'end'\n"
            "- If documents are available, prefer 'rag' route to search them first"
        )
        messages = [SystemMessage(content=system_prompt)] + state["messages"]

        try:
            result_str = router_llm.invoke(messages)
            result_dict = json.loads(result_str.strip())
            route = result_dict.get("route", "answer")
            reply = result_dict.get("reply")

            if state.get("document_ids") and route != "end":
                route = "rag"

            out = {"messages": state["messages"], "route": route}
            if route == "end":
                out["messages"] = state["messages"] + [AIMessage(content=reply or "Hello! How can I help you today?")]

            if state.get("document_ids"):
                out["document_ids"] = state["document_ids"]

            return out
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing router response: {e}")
            route = "rag" if state.get("document_ids") else "answer"
            return {
                "messages": state["messages"],
                "route": route,
                "document_ids": state.get("document_ids", [])
            }

    def rag_node(self, state: AgentState) -> AgentState:
        """Enhanced RAG node with document-specific search"""
        query = next((m.content for m in reversed(state["messages"])
                      if isinstance(m, HumanMessage)), "")

        document_ids = state.get("document_ids")
        chunks = rag_search_tool.invoke({
            "query": query,
            "document_ids": document_ids
        })

        document_context = ""
        if document_ids:
            document_context = f"The user has uploaded {len(document_ids)} specific document(s) for analysis. "

        judge_prompt = (
            f"You are evaluating if retrieved information is sufficient to answer the user's question. "
            f"{document_context}"
            f"Respond with ONLY 'YES' if sufficient or 'NO' if not sufficient.\n\n"
            f"Question: {query}\n\nRetrieved info: {chunks}\n\n"
            f"Is this sufficient to answer the question? "
            f"Consider: if user uploaded documents and we found relevant content, that's usually sufficient."
        )

        try:
            verdict_str = judge_llm.invoke([HumanMessage(content=judge_prompt)])
            sufficient = verdict_str.strip().upper() == 'YES'

            # If we have documents but no good context, still try web search
            # If we have good context from documents, use it
            next_route = "answer" if sufficient else "web"

            return {
                **state,
                "rag": chunks,
                "route": next_route
            }
        except Exception as e:
            logger.error(f"Error in judge: {e}")
            # If we have document context, use it; otherwise try web
            next_route = "answer" if chunks and not chunks.startswith("No relevant") else "web"
            return {
                **state,
                "rag": chunks,
                "route": next_route
            }

    # ── Node 3: web search ───────────────────────────────────────────────
    def web_node(self, state: AgentState) -> AgentState:
        """Enhanced web search node"""
        query = next((m.content for m in reversed(state["messages"])
                      if isinstance(m, HumanMessage)), "")
        snippets = web_search_tool.invoke({"query": query})
        return {**state, "web": snippets, "route": "answer"}

    # ── Node 4: final answer ─────────────────────────────────────────────
    def answer_node(self, state: AgentState) -> AgentState:
        user_q = next((m.content for m in reversed(state["messages"])
                       if isinstance(m, HumanMessage)), "")

        ctx_parts = []
        source_info = []

        if state.get("rag") and not state["rag"].startswith("No relevant"):
            ctx_parts.append("📄 **Information from your uploaded documents:**\n" + state["rag"])
            source_info.append("uploaded documents")

        if state.get("web"):
            ctx_parts.append("🌐 **Additional information from web search:**\n" + state["web"])
            source_info.append("web search")

        context = "\n\n".join(ctx_parts) if ctx_parts else "No external context available."

        system_prompt = """You are an intelligent assistant that provides comprehensive and well-formatted responses.

        RESPONSE GUIDELINES:
        1. **Use the provided context effectively** - synthesize information from documents and web sources
        2. **Be specific and detailed** when context is available
        3. **Format your response clearly** with headers, bullet points, or sections when appropriate
        4. **Cite your sources** - mention when information comes from uploaded documents vs web search
        5. **If you used web search** because documents didn't have the info, explain this clearly
        6. **Be conversational but professional**

        CONTEXT SOURCES:
        """ + (f"- Information from: {', '.join(source_info)}" if source_info else "- No external sources available")

        user_prompt = f"""Question: {user_q}

        Context:
        {context}

        Please provide a comprehensive response based on the available information. If you found information through web search because it wasn't in the uploaded documents, please mention this clearly."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        try:
            ans = answer_llm.invoke(messages).content

            if len(source_info) > 1:
                ans += f"\n\n*Sources: {', '.join(source_info)}*"
            elif state.get("web") and not state.get("rag"):
                ans += "\n\n*Note: This information was found through web search as it wasn't available in your uploaded documents.*"

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            ans = "I apologize, but I encountered an error while generating the response. Please try rephrasing your question."

        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=ans)]
        }


# ── Routing helpers ─────────────────────────────────────────────────
def from_router(st: AgentState) -> Literal["rag", "answer", "end"]:
    return st["route"]


def after_rag(st: AgentState) -> Literal["answer", "web"]:
    return st["route"]


def after_web(_) -> Literal["answer"]:
    return "answer"


class RAGGraph:
    def __init__(self, azure_api_key: str = None, azure_endpoint: str = None,
                 azure_api_version: str = None, openai_api_key: str = None):

        self.nodes = RAGNodes(
            azure_api_key, azure_endpoint,
            azure_api_version, openai_api_key
        )

        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow"""
        # ── Build graph ─────────────────────────────────────────────────────
        workflow = StateGraph(AgentState)

        workflow.add_node("router", self.nodes.router_node)
        workflow.add_node("rag_lookup", self.nodes.rag_node)
        workflow.add_node("web_search", self.nodes.web_node)
        workflow.add_node("answer", self.nodes.answer_node)

        workflow.set_entry_point("router")

        workflow.add_conditional_edges("router", from_router,
                                       {"rag": "rag_lookup", "answer": "answer", "end": END})
        workflow.add_conditional_edges("rag_lookup", after_rag,
                                       {"answer": "answer", "web": "web_search"})
        workflow.add_edge("web_search", "answer")
        workflow.add_edge("answer", END)

        memory = MemorySaver()
        agent = workflow.compile(checkpointer=memory)

        return agent

    def invoke(self, initial_state: dict, config: dict = None) -> dict:
        """Invoke the DataDialect workflow"""
        try:
            if not isinstance(initial_state, dict):
                raise ValueError("initial_state must be a dictionary")

            if config is None:
                config = {"configurable": {"thread_id": "default"}}

            result = self.graph.invoke(initial_state, config=config)

            if result.get("error"):
                logger.error(f"Error in RAG workflow: {result['error']}")
                return {
                    "success": False,
                    "error": result["error"],
                    "messages": result.get("messages", [])
                }

            return {
                "success": True,
                "messages": result.get("messages", []),
                "route": result.get("route"),
                "rag": result.get("rag"),
                "web": result.get("web"),
                "document_ids": result.get("document_ids", [])
            }

        except Exception as e:
            logger.error(f"Error in LangGraph execution: {e}")
            return {
                "success": False,
                "error": f"Workflow execution failed: {str(e)}",
                "messages": initial_state.get("messages", [])
            }


def create_rag_graph():
    """Factory function to create RAGGraph instance"""
    return RAGGraph(
        azure_api_key=azure_api_key,
        azure_endpoint=azure_endpoint,
        azure_api_version=azure_api_version,
        openai_api_key=openai_api_key
    )


rag_graph_instance = create_rag_graph()