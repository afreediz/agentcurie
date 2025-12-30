# type: ignore
from typing import Dict, Type

from pydantic import BaseModel, Field, create_model, RootModel
from .views import (
	AgentsRegistry,
	RegisteredAgent,
	AgentCard
)
from .views import BaseAgent, RegisteredAgent
from .views import AgentModel

class Registry:
	"""Service for registering and managing agents"""

	def __init__(self):
		self.registry = AgentsRegistry()
		self.active_agents:Dict[str, BaseAgent] = {}

	def register_agent(self, card: AgentCard, agent_class: Type[BaseAgent],):
		"""function for registering agents"""
		self.registry.agents[card.name] = RegisteredAgent(card=card, agent_class=agent_class)

	def is_agent_exist(self, agent_name: str):
		if agent_name in self.registry.agents:
			return True
		
		return False

	def get_prompt_description(self) -> str:
		"""Get a description of all agents for the prompt"""
		return self.registry.get_prompt_description()
	
	def create_agent_instance(self, agent_name: str):
		if agent_name not in self.registry.agents:
			raise ValueError(f"Unknow agent type: {agent_name}")
		
		agent_class = self.registry.agents[agent_name].agent_class
		agent_instance = agent_class()
		agent_instance.agent_name = agent_name
		return agent_instance
	
	def get_agent_instance(self, agent_name) -> BaseAgent:
		if agent_name in self.active_agents:
			print("Getting from pre-initialized instances")
			return self.active_agents[agent_name]
		else:
			agent = self.create_agent_instance(agent_name)
			if self.registry.agents[agent_name].card.persistent:
				self.active_agents[agent_name] = agent

			return agent
		
	def delete_active_agent(self, agent_name) -> None:
		if agent_name in self.active_agents:
			del self.active_agents

		return
	
	def create_agent_model(self, exclude_agents:list[str] = []) -> Type[AgentModel]:
		from typing import Union
		
		individual_agent_models:list[Type[BaseModel]] = []

		# dynamic param or agent guided param initialization no added yet, will implement in future
		for agent_name, _ in self.registry.agents.items():
			if agent_name in exclude_agents:
				continue
			
			individual_agent_model = create_model(
				f'{agent_name.title().replace('_','')}AgentModel',
				__base__=AgentModel,
				**{
					agent_name:(
						str,
						Field(..., description=f'message you want to pass to {agent_name}')
					)
				}
			)
			individual_agent_models.append(individual_agent_model)

		if not individual_agent_models:
			return create_model('EmptyAgentModel', __base__=AgentModel)

		# Create proper Union type that maintains ToolModel interface
		if len(individual_agent_models) == 1:
			result_model = individual_agent_models[0]
		else:
			# Create a Union type using RootModel that properly delegates ToolModel methods
			union_type = Union[tuple(individual_agent_models)]

			class AgentModelUnion(RootModel[union_type]):
				"""Union of all available tool models that maintains ToolModel interface"""

				def model_dump(self, **kwargs):
					"""Delegate model_dump to the underlying tool model"""
					if hasattr(self.root, 'model_dump'):
						return self.root.model_dump(**kwargs) #type:ginore
					return super().model_dump(**kwargs)
				
				def get_type(self):
					return self.root.get_type()

			# Set the name for better debugging
			AgentModelUnion.__name__ = 'AgentModel'
			AgentModelUnion.__qualname__ = 'AgentModel'

			result_model = AgentModelUnion

		return result_model