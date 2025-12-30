# TOOLS CONTROLLER LOGIC
from pydantic import BaseModel
from typing import TypeVar
from .registery.service import Registry
from .views import ToolModel, ToolResult
from .tool_views import Done

T = TypeVar('T', bound=BaseModel)

class Controller():
	def __init__(
		self,
		exclude_tools: list[str] = [],
		include_done_tool:bool = True
	):
		self.exclude_tools = exclude_tools
		self.registry = Registry(exclude_tools)

		if include_done_tool:
			@self.registry.tool("Complete task - provide a summary of results for the user. Set success=True if task completed successfully, false otherwise. Text should be your response to the user summarizing results.", param_model=Done)
			async def done(params: Done) -> ToolResult:
				return ToolResult(content=params.summary, is_done=True)

	def tool(self, description: str, **kwargs):
		"""Decorator for registering custom tools

		@param description: Describe the LLM what the function does (better description == better function calling)
		"""
		return self.registry.tool(description, **kwargs)
	
	async def act(self, tool: ToolModel) -> ToolResult:
		"""Execute a tool"""
		try:
			for tool_name, params in tool.model_dump(exclude_unset=True).items():
				if params is not None:
					# remove highlights
					result = await self.registry.execute_tool(tool_name, params)
					if isinstance(result, str):
						return ToolResult(content=result)
					elif isinstance(result, ToolResult):
						return result
					elif result is None:
						return ToolResult()
					else:
						raise ValueError(f'Invalid tool result type: {type(result)} of {result}')
			return ToolResult()
		except Exception as e:
			raise e
