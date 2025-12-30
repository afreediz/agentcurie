import uuid
import asyncio
from typing import Dict, Any, Optional, Literal, Callable, Awaitable, TYPE_CHECKING
from pydantic import BaseModel, ConfigDict, Field, create_model
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from langchain_core.messages.base import BaseMessage

if TYPE_CHECKING:
    from supervisor import SupervisorAgent
    from controller import BaseAgent

# Event Types
class EventType(Enum):
    SOLVE_REQUEST_SUPERVISOR = "solve_request_supervisor"
    SOLVE_REQUEST_CHILD_AGENT = "solve_request_child_agent"
    QUERY_REQUEST = "query_request"
    CHILD_RESPONSE = "child_response"
    RESPONSE = "response"
    ERROR = "error"

@dataclass
class Event:
    event_type: EventType
    payload: Dict[str, Any]
    message: Optional[str] = None
    id: str = uuid.uuid4().hex
    task_id: Optional[str] = None
    agent_name: Optional[str] = None
    timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time())

class AgentStatus(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING_FOR_QUERY = "waiting_for_query"
    QUERY_PROCESSED = "query_processed"
    QUERY_INTERCEPTED = "query_intercepted"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class TaskContext:
    message: str
    id: str = uuid.uuid4().hex
    status: AgentStatus = AgentStatus.IDLE
    result: Optional[Any] = None
    error: Optional[str] = None

@dataclass
class AgentContext:
    agent_name: str
    message: str
    id: str = uuid.uuid4().hex
    query: Optional[str] = None
    status: AgentStatus = AgentStatus.IDLE
    instance_id: str  = uuid.uuid4().hex
    result: Optional[str] = None
    error: Optional[str] = None

@dataclass
class QueryMessage:
    agent_name: str
    message: str
    id: str = uuid.uuid4().hex

class AgentResult(BaseModel):
    content:str|None = None
    error:str|None = None
    messages:list[BaseMessage] = []
    resources:dict = {}

class Hook(BaseModel):
    order: Literal['before', 'after']

class FuncHook(Hook):
    """Executes given func with supervisor"""
    func: Callable[['SupervisorAgent'], Awaitable]

class AgentHook(FuncHook):
    """Passes given agent as first param to func along with supervisor"""
    agent_name: str
    func: Callable[['BaseAgent','SupervisorAgent'], Awaitable]