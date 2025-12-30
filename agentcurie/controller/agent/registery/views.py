from typing import Callable, Dict, Type, Optional, Literal
from pydantic import BaseModel, ConfigDict, Field, RootModel
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Type
import uuid

class AgentModel(BaseModel):
    """Base model for dynamically created agent models"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
     
    def get_type(self) -> Literal['agent']:
        return "agent"
	
class AgentCard(BaseModel):
    """Defines agent information and skills"""
    name: str = Field(..., description="Your agent name, this will be used to uniquel access agent accross the system")
    description: str = Field(..., description="Describe what your agent does")
    skills: list[str] = Field(..., description="Descirbe set of skills your agent can perform")
    persistent: Optional[bool] = False

class SuperVisor(ABC):

    @abstractmethod
    async def _solve_query(self, message, agent_name) -> str:
        """Resolve query from child agents"""
        raise NotImplementedError('_solve_query must be overwrited')

class BaseAgent(ABC):
    def __init__(self):
        """
         BaseAgent from which All agents must inherit from.
         initialize your agent here and call the agent inside process function
        """
        self.agent_name:str|None = None # assigned in runtime
        self.supervisor:SuperVisor # assigned in runtime
        self.uid = uuid.uuid4().hex

        # tool = Tool(func=self.query_supervisor, desc="...")
        # self.agent = YourAgent(tools=[tool, ...])
        
    def set_supervisor(self, supervisor):
        self.supervisor = supervisor
    
    @abstractmethod
    async def process(self, message: str) -> Any:
        """Process the main task - should yield control back when needing to query"""
        # self.agent.invoke(message)
        raise RuntimeError("Not overrided")
    
    async def query_supervisor(self, query: str) -> str:
        """Tool to query the supervisor for help. If you stuck on any problem or need assistance, use this tool"""
        if not self.supervisor:
            raise RuntimeError("No supervisor set")

        if not self.agent_name:
            raise RuntimeError("Agent name not configured, SuperVisor Error")

        # message = QueryMessage(agent_id=self.agent_name, message=query)
        result = await self.supervisor._solve_query(query, self.agent_name)
        
        return result

class RegisteredAgent(BaseModel):
    card: AgentCard
    agent_class: Type[BaseAgent]

    def prompt_description(self):
        s = f'{self.card.name}: '
        s += f'{self.card.description}.'
        s += f' Skills: {', '.join(self.card.skills)}.'
        return s

class AgentsRegistry(BaseModel):
    """Model representing the action registry"""

    # name: card - for quick access
    agents: Dict[str, RegisteredAgent] = {}

    def get_prompt_description(self) -> str:
        """Get a description of all actions for the prompt"""
        return '\n'.join([agent.prompt_description() for agent in self.agents.values()])