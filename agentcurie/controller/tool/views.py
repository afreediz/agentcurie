from pydantic import BaseModel, ConfigDict, create_model, Field
from typing import Optional
from .registery.views import ToolModel

class ToolResult(BaseModel):
    """Result of executing an action"""		
    is_done: bool | None = False
    success: bool | None = True
    error: str | None = None
    content: str | None = None

# dynamically type guided model invoking schema for accurate tool params
class AgentOutput(BaseModel):
	model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

	evaluation_previous_goal: str
	memory: str
	next_goal: str
	choice: ToolModel = Field(
		...,
		description='tool to execute',
	)

	@staticmethod
	def type_with_custom_tools(tools_model: type[ToolModel]) -> type['AgentOutput']:
		"""Extend actions with custom actions"""

		model_ = create_model(
			'AgentOutput',
			__base__=AgentOutput,
			choice=(
				tools_model,  # type: ignore
				Field(..., description='tool to execute'),
			),
			__module__=AgentOutput.__module__,
		)
		model_.__doc__ = 'AgentOutput model with custom actions'
		return model_