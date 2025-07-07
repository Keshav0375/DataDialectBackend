import asyncio
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.models.sample_chat_state import ChatState
from app.nodes.router_node import RouterNode
from app.nodes.sample_SQL_Node import SQLNode
from app.nodes.document_node import DocumentNode
from app.nodes.nosql_node import NoSQLNode
from app.nodes.response_node import ResponseNode


class GraphManager:
    """
    Advanced LangGraph manager with optimized execution and state management.
    Handles routing, execution, and response generation with high performance.
    """

    def __init__(self):
        self.graph = None
        self.checkpointer = MemorySaver()
        self.semaphore = asyncio.Semaphore(50)

        # Initialize node processors
        self.router_node = RouterNode()
        self.sql_node = SQLNode()
        self.document_node = DocumentNode()
        self.nosql_node = NoSQLNode()
        self.response_node = ResponseNode()

    async def initialize(self):
        """Initialize the LangGraph with optimized routing logic."""
        # Initialize nodes
        await self.sql_node.initialize()
        await self.document_node.initialize()
        await self.nosql_node.initialize()

        # Create state graph
        workflow = StateGraph(ChatState)

        # Add nodes
        workflow.add_node("router", self.router_node.process)
        workflow.add_node("sql_handler", self.sql_node.process)
        workflow.add_node("document_handler", self.document_node.process)
        workflow.add_node("nosql_handler", self.nosql_node.process)
        workflow.add_node("response_formatter", self.response_node.process)

        # Set entry point
        workflow.set_entry_point("router")

        # Add conditional routing
        workflow.add_conditional_edges(
            "router",
            self._route_to_handler,
            {
                "sql": "sql_handler",
                "document": "document_handler",
                "nosql": "nosql_handler",
                "error": "response_formatter"
            }
        )

        # All handlers go to response formatter
        workflow.add_edge("sql_handler", "response_formatter")
        workflow.add_edge("document_handler", "response_formatter")
        workflow.add_edge("nosql_handler", "response_formatter")

        # Response formatter ends the workflow
        workflow.add_edge("response_formatter", END)

        # Compile graph
        self.graph = workflow.compile(checkpointer=self.checkpointer)

    def _route_to_handler(self, state: ChatState) -> str:
        """Advanced routing logic based on intent detection."""
        if state.error:
            return "error"

        intent = state.detected_intent
        confidence = state.confidence_score

        if confidence < 0.6:
            state.error = "Unable to determine intent with sufficient confidence"
            return "error"

        return intent

    async def process_chat(self, user_message: str, chat_history: list = None) -> Dict[str, Any]:
        """Process chat message with optimized execution."""
        async with self.semaphore:
            try:
                # Create initial state
                initial_state = ChatState(
                    user_message=user_message,
                    chat_history=chat_history or [],
                    detected_intent="",
                    confidence_score=0.0,
                    processed_data=None,
                    response="",
                    error=None,
                    metadata={}
                )

                # Execute graph with timeout
                result = await asyncio.wait_for(
                    self.graph.ainvoke(initial_state, {"configurable": {"thread_id": "default"}}),
                    timeout=30.0
                )

                return {
                    "response": result.response,
                    "intent": result.detected_intent,
                    "confidence": result.confidence_score,
                    "metadata": result.metadata,
                    "error": result.error
                }

            except asyncio.TimeoutError:
                return {
                    "response": "Request timeout - please try again",
                    "error": "timeout"
                }
            except Exception as e:
                return {
                    "response": "Sorry, I encountered an error processing your request.",
                    "error": str(e)
                }

    async def close(self):
        """Cleanup resources."""
        await self.sql_node.close()
        await self.document_node.close()
        await self.nosql_node.close()