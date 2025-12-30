from typing import Callable, Dict, Type, Literal
from pydantic import BaseModel, ConfigDict

class ToolModel(BaseModel):
    """Base model for dynamically created tool models"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
     
    def get_type(self) -> Literal['tool']:
        return "tool"
	
class RegisteredTool(BaseModel):
	"""Model for a registered action"""

	name: str
	description: str
	function: Callable
	param_model: Type[BaseModel]

	model_config = ConfigDict(arbitrary_types_allowed=True)

	def prompt_description(self) -> str:
		"""Get a description of the action for the prompt"""
		skip_keys = ['title']
		s = f'{self.description}: \n'
		s += '{' + str(self.name) + ': '
		s += str(
			{
				k: {sub_k: sub_v for sub_k, sub_v in v.items() if sub_k not in skip_keys}
				for k, v in self.param_model.schema()['properties'].items()
			}
		)
		s += '}'
		return s


class ToolsRegistry(BaseModel):
	"""Model representing the action registry"""

	tools: Dict[str, RegisteredTool] = {}

	def get_prompt_description(self) -> str:
		"""Get a description of all actions for the prompt"""
		return '\n'.join([tool.prompt_description() for tool in self.tools.values()])
