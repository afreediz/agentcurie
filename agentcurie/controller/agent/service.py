# AGENT CONTROLLER LOGIC
from pydantic import BaseModel
from typing import Any, TypeVar, Type, Optional
from .registery.service import Registry
from .views import AgentResult
from .registery.views import AgentCard, AgentModel, BaseAgent
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

T = TypeVar('T', bound=BaseModel)

class Controller():
	def __init__(
		self,
		supervisor,
		include_done_agent:bool = False,
		llm:Optional[BaseChatOpenAI] = None
	):
		self.supervisor = supervisor
		self.registry = Registry()

		if include_done_agent:
			if llm is None:
				raise ValueError("llm must needed to register doneAgent")
			class Summary(BaseModel):
				summary: str

			card = AgentCard(name="done_agent", description="Finish agents execution with summary to provide user", skills=['provides users comprehensive summary'])
			class Done(BaseAgent):
				def __init__(self):
					self.llm = llm
					
				async def process(self, message: str) -> Any:
					sys_message = SystemMessage(content="Your good at summarising the results and providing user a good insight")
					res:Summary = await self.llm.with_structured_output(Summary).ainvoke([sys_message, HumanMessage(message)]) #type:ignore

					return res.summary
				
			self.registry.register_agent(card=card, agent_class=Done)

	def register_agent(self, card: AgentCard, agent_class: Type[BaseAgent]):
		"""Decorator for registering custom agent

		@param description: Describe the LLM what the function does (better description == better function calling)
		"""
		return self.registry.register_agent(card, agent_class)
	
	async def act(self, agent: AgentModel) -> AgentResult:
		try:
			for agent_name, message in agent.model_dump(exclude_unset=True).items():

				if not self.registry.is_agent_exist(agent_name):
					raise ValueError(f"Unknow agent : {agent_name}")
				
				instance = self.registry.get_agent_instance(agent_name)
				instance.set_supervisor(self.supervisor)
				# if instance.supervisor is None:
					# instance.set_supervisor(self.supervisor)

				res = await instance.process(message)
				return AgentResult(content=res)
			return AgentResult()
		except Exception as e:
			raise e
			return AgentResult(error=str(e))
