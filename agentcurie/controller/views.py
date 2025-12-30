from pydantic import BaseModel, ConfigDict, RootModel, Field, create_model
from .tool.registery.views import ToolModel
from .agent.registery.views import AgentModel
from typing import Union, Type, Literal

import logging
logger = logging.getLogger(__name__)

class ChoiceModel(BaseModel):
    choice: RootModel[Union[(ToolModel, AgentModel)]]
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def get_type(self) -> Literal['tool', 'agent']:
        return self.root.get_type() #type:ignore
    
# dynamically type guided model invoking schema for accurate tool & agent params.
class AgentOutput(BaseModel):
	model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

	evaluation_previous_goal: str
	memory: str
	next_goal: str
	action: ChoiceModel = Field(
		...,
		description='tool or agent to execute',
	)

	@staticmethod
	def type_with_custom_tools_and_agents(choice_model: type[ChoiceModel]) -> type['AgentOutput']:
		"""Extend actions with custom actions"""

		model_ = create_model(
			'AgentOutput',
			__base__=AgentOutput,
			action=(
				choice_model,  # type: ignore
				Field(..., description='tool or agent to execute'),
			),
			__module__=AgentOutput.__module__,
		)
		model_.__doc__ = 'AgentOutput model with custom actions'
		return model_
      
	def get_choice(self) -> Literal['tool','agent']:
		try:
			try:
				choice = self.action.choice.root.get_type()
			except Exception as e:
				choice = self.action.choice.get_type() #type:ignore
				
			return choice
		except Exception as e:
			logger.info(e)
			raise e

class ChoiceResult(BaseModel):
	"""Result of executing an agent"""		
	is_done: bool | None = False
	success: bool | None = True
	error: str | None = None
	content: str | None = None
    
	def has_errors(self):
		if self.error is not None:
			return True
		
		return False