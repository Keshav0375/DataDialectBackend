from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class ChatState(BaseModel):
    """Comprehensive state model for chat processing pipeline."""

    # Input data
    user_message: str = Field(..., description="Current user message")
    chat_history: List[Dict[str, Any]] = Field(default_factory=list, description="Previous chat messages")

    # Routing information
    detected_intent: str = Field(default="", description="Detected user intent (sql/document/nosql)")
    confidence_score: float = Field(default=0.0, description="Confidence score for intent detection")

    # Processing data
    processed_data: Optional[Dict[str, Any]] = Field(default=None, description="Processed data from handlers")

    # Output
    response: str = Field(default="", description="Generated response")
    error: Optional[str] = Field(default=None, description="Error message if any")

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        arbitrary_types_allowed = True