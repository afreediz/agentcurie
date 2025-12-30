from pydantic import BaseModel, ConfigDict, Field, create_model
from .registery.views import AgentModel

class AgentResult(BaseModel):
    """Result of executing an agent"""		
    is_done: bool | None = False
    success: bool | None = True
    error: str | None = None
    content: str | None = None

class AgentOutput(BaseModel):
	model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

	evaluation_previous_goal: str
	memory: str
	next_goal: str
	choice: AgentModel = Field(
		...,
		description='agent to execute',
	)

	@staticmethod
	def type_with_custom_agents(agents_model: type[AgentModel]) -> type['AgentOutput']:
		"""Extend actions with custom actions"""

		model_ = create_model(
			'AgentOutput',
			__base__=AgentOutput,
			choice=(
				agents_model,  # type: ignore
				Field(..., description='agent to execute'),
			),
			__module__=AgentOutput.__module__,
		)
		model_.__doc__ = 'AgentOutput model with custom actions'
		return model_