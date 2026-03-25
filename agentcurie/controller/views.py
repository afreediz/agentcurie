from pydantic import BaseModel, ConfigDict, Field, create_model
from .tool.registery.views import ToolModel
from .agent.registery.views import AgentModel
from typing import Any, List, Union, Type, Literal

import logging
logger = logging.getLogger(__name__)


class ToolsAction(BaseModel):
    """Execute one or more tools in sequence"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    tools: List[Any]  # list[ToolModel] — typed precisely in dynamic subclass

    def get_type(self) -> Literal['tool']:
        return 'tool'


class AgentAction(BaseModel):
    """Delegate to exactly one agent"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    agent: Any  # AgentModel — typed precisely in dynamic subclass

    def get_type(self) -> Literal['agent']:
        return 'agent'


class ChoiceModel(BaseModel):
    """Holds either a ToolsAction (one or more tools) or an AgentAction (one agent)"""
    choice: Union[ToolsAction, AgentAction]
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_type(self) -> Literal['tool', 'agent']:
        return self.choice.get_type()


# dynamically type guided model invoking schema for accurate tool & agent params.
class AgentOutput(BaseModel):
	model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

	evaluation_previous_goal: str
	memory: str
	next_goal: str
	action: ChoiceModel = Field(
		...,
		description='tools (one or more) or a single agent to execute',
	)

	@staticmethod
	def type_with_custom_tools_and_agents(choice_model: type[ChoiceModel]) -> type['AgentOutput']:
		"""Extend actions with custom actions"""

		model_ = create_model(
			'AgentOutput',
			__base__=AgentOutput,
			action=(
				choice_model,  # type: ignore
				Field(..., description='tools (one or more) or a single agent to execute'),
			),
			__module__=AgentOutput.__module__,
		)
		model_.__doc__ = 'AgentOutput model with custom actions'
		return model_

	def get_choice(self) -> Literal['tool', 'agent']:
		return self.action.choice.get_type()


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
