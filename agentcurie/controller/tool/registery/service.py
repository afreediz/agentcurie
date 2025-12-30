# type: ignore
import asyncio
from inspect import iscoroutinefunction, signature
from typing import Any, Callable, Optional, Type, Literal

from pydantic import BaseModel, Field, create_model, RootModel
from .views import (
	ToolsRegistry,
	RegisteredTool,
)
from agentcurie.controller.tool.views import ToolModel

class Registry:
	"""Service for registering and managing tools"""

	def __init__(self, exclude_tools: list[str] = []):
		self.registry = ToolsRegistry()
		self.exclude_tools = exclude_tools

	def _create_param_model(self, function: Callable) -> Type[BaseModel]:
		"""Creates a Pydantic model from function signature"""
		sig = signature(function)
		params = {
			name: (param.annotation, ... if param.default == param.empty else param.default)
			for name, param in sig.parameters.items()
		}
		# TODO: make the types here work
		return create_model(
			f'{function.__name__}_parameters',
			__base__=ToolModel,
			**params,  # type: ignore
		)

	def tool(
		self,
		description: str,
		param_model: Optional[Type[BaseModel]] = None,
	):
		"""Decorator for registering tools"""

		def decorator(func: Callable):
			# Skip registration if tool is in exclude_tools
			if func.__name__ in self.exclude_tools:
				return func

			# Create param model from function if not provided
			actual_param_model = param_model or self._create_param_model(func)

			# Wrap sync functions to make them async
			if not iscoroutinefunction(func):

				async def async_wrapper(*args, **kwargs):
					return await asyncio.to_thread(func, *args, **kwargs)

				# Copy the signature and other metadata from the original function
				async_wrapper.__signature__ = signature(func)
				async_wrapper.__name__ = func.__name__
				async_wrapper.__annotations__ = func.__annotations__
				wrapped_func = async_wrapper
			else:
				wrapped_func = func

			tool = RegisteredTool(
				name=func.__name__,
				description=description,
				function=wrapped_func,
				param_model=actual_param_model,
			)
			self.registry.tools[func.__name__] = tool
			return func

		return decorator

	async def execute_tool(self, tool_name: str, params: dict) -> Any:
		"""Execute a registered tool"""
		if tool_name not in self.registry.tools:
			raise ValueError(f'tool {tool_name} not found')

		tool = self.registry.tools[tool_name]
		try:
			# Create the validated Pydantic model
			validated_params = tool.param_model(**params)

			# Check if the first parameter is a Pydantic model
			sig = signature(tool.function)
			parameters = list(sig.parameters.values())
			is_pydantic = parameters and issubclass(parameters[0].annotation, BaseModel)

			if is_pydantic:
				return await tool.function(validated_params)
			
			return await tool.function(**validated_params.model_dump())

		except Exception as e:
			raise RuntimeError(f'Error executing tool {tool_name}: {str(e)}') from e

	def create_tool_model(self, exclude_tools: list[str]=[]) -> type[ToolModel]:
		"""Creates a Union of individual tool models from registered tool,
		used by LLM APIs that support tool calling & enforce a schema.
		"""
		from typing import Union

		available_tools: dict[str, RegisteredTool] = {}
		for name, tool in self.registry.tools.items():
			if name in exclude_tools:
				continue

			available_tools[name] = tool

		# Create individual tool models for each tools
		individual_tool_models: list[Type[BaseModel]] = []

		# print("AVAILABLE TOOLS : ", " ,".join([key for (key, value) in available_tools.items()]))
		for name, tool in available_tools.items():
			# Create an individual model for each tool that contains only one field
			individual_model = create_model(
				f'{name.title().replace("_", "")}ToolModel',
				__base__=ToolModel,
				**{
					name: (
						tool.param_model,
						Field(..., description=tool.description),
					)
				},
			)
			individual_tool_models.append(individual_model)

		if not individual_tool_models:
			return create_model('EmptyToolModel', __base__=ToolModel)

		# Create proper Union type that maintains ToolModel interface
		if len(individual_tool_models) == 1:
			result_model = individual_tool_models[0]

		else:
			# Create a Union type using RootModel that properly delegates ToolModel methods
			union_type = Union[tuple(individual_tool_models)]

			class ToolModelUnion(RootModel[union_type]):
				"""Union of all available tool models that maintains ToolModel interface"""

				def model_dump(self, **kwargs):
					"""Delegate model_dump to the underlying tool model"""
					if hasattr(self.root, 'model_dump'):
						return self.root.model_dump(**kwargs) #type:ginore
					return super().model_dump(**kwargs)
				
				def get_type(self) -> Literal['tool']:
					return 'tool'

			# Set the name for better debugging
			ToolModelUnion.__name__ = 'ToolModel'
			ToolModelUnion.__qualname__ = 'ToolModel'

			result_model = ToolModelUnion

		return result_model  # type:ignore

	def get_prompt_description(self) -> str:
		"""Get a description of all tools for the prompt"""
		return self.registry.get_prompt_description()
