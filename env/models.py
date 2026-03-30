from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict, Any


class Message(BaseModel):
    role: Literal["user", "agent"]
    content: str


class Observation(BaseModel):
    ticket_id: int
    conversation: List[Message]
    category: Optional[str]
    priority: Optional[str]
    status: Literal["open", "in_progress", "escalated", "closed"]
    steps_remaining: int
    sla_remaining: int
    tool_result: Optional[Dict[str, Any]] = None


class Action(BaseModel):
    action_type: Literal[
        "classify", "prioritize", "respond",
        "escalate", "resolve",
        "refund_api", "db_lookup"
    ]
    content: Optional[str] = None


class Reward(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    breakdown: Dict[str, float]