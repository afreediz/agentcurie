from __future__ import annotations

import json
from typing import List, Optional
from pydantic import BaseModel
from langchain_core.messages import (
	AIMessage,
	BaseMessage,
	HumanMessage,
	SystemMessage,
	ToolMessage,
)

from .views import MessageHistory, MessageMetadata
from agentcurie.supervisor.prompts import SystemPrompt
from agentcurie.controller import AgentOutput
from agentcurie.supervisor.utils import get_first_key_param

import logging
logger = logging.getLogger(__name__)


class MessageManager:
	def __init__(
		self,
		task: str|None = None,
		max_input_tokens: int = 128000,
		max_error_length: int = 400,
		estimated_characters_per_token: int = 3,
		tools_and_agents_description: str|None = None,
		system_prompt: SystemPrompt|None = None,
		message_context: Optional[str] = None,
	):
		self.max_input_tokens = max_input_tokens
		self.history = MessageHistory()
		self.task = task
		self.estimated_characters_per_token = estimated_characters_per_token
		self.max_error_length = max_error_length
		self.message_context = message_context
		self.IMG_TOKENS = 10

		system_prompt = system_prompt if system_prompt is not None else SystemPrompt()

		if tools_and_agents_description:
			system_message = system_prompt.get_system_message(tools_and_agents_description)

			self.system_prompt = system_message
			self._add_message_with_tokens(system_message)

		if self.message_context:
			context_message = HumanMessage(content=self.message_context)
			self._add_message_with_tokens(context_message)

		if task:
			task_message = self.task_instructions(task)
			self._add_message_with_tokens(task_message)

	@staticmethod
	def task_instructions(task: str) -> HumanMessage:
		content = f'Your task is: {task}.'
		return HumanMessage(content=content)

	@staticmethod
	def task_instructions_raw(task: str) -> str:
		return f'Your task is: {task}.'

	def add_new_task(self, new_task: str) -> None:
		self.task = new_task
		content = (
			f'Your task is: {new_task}.'
		)
		msg = HumanMessage(content=content)
		self._add_message_with_tokens(msg)

	# Guide the model through states, in future if needed
	# def add_state_message(
	# 	self,
	# 	result: Optional[List[ActionResult]] = None,
	# 	step_info: Optional[AgentStepInfo] = None,
	# ) -> None:
	# 	"""Add browser state as human message"""

	# 	# if keep in memory, add to directly to history and add state without result
	# 	if result:
	# 		for r in result:
	# 			if r.include_in_memory:
	# 				if r.extracted_content:
	# 					msg = HumanMessage(content='Action result: ' + str(r.extracted_content))
	# 					self._add_message_with_tokens(msg)
	# 				if r.error:
	# 					msg = HumanMessage(content='Action error: ' + str(r.error)[-self.max_error_length :])
	# 					self._add_message_with_tokens(msg)
	# 				result = None  # if result in history, we dont want to add it again

	# 	# otherwise add state message and result to next message (which will not stay in memory)
	# 	state_message = AgentMessagePrompt(
	# 		state,
	# 		result,
	# 		include_attributes=self.include_attributes,
	# 		max_error_length=self.max_error_length,
	# 		step_info=step_info,
	# 	).get_user_message()
	# 	self._add_message_with_tokens(state_message)

	def add_ai_message(self, message:str) -> None:
		self._add_message_with_tokens(
			AIMessage(content=message)
		)

	def add_human_message(self, message:str) -> None:
		self._add_message_with_tokens(
			HumanMessage(content=message)
		)

	def add_raw_tool_message(self, message: str) -> None:
		self._add_message_with_tokens(
			ToolMessage(
				content=message,
			)
		)

	def add_tool_message(self, message:str, name:str) -> None:
		self._add_message_with_tokens(
			ToolMessage(
				content=message,
				name=name,
				tool_call_id=name
			)
		)

	def pretty_print_messages(self) -> None:
		logger.info('------------------------------------------------------------------------------------------------')
		for message in self.history.messages[2:]:
			message = message.message
			if isinstance(message, ToolMessage):
				logger.info(f"TOOL({message.name}): {message.content}")
			elif isinstance(message, AIMessage):
				logger.info(f"AI: {message.content}, calling tools: {', '.join([m.get('name') for m in message.tool_calls])}")
			elif isinstance(message, HumanMessage):
				logger.info(f"HUMAN: {message.content}")
			else:
				logger.info(f"{message.content}")
		logger.info('------------------------------------------------------------------------------------------------')

	def get_key_param(self, model: dict):
		for key, value in model.items():
			key_name = key
			param = value
			break

		return key_name, param

	def format_agentoutput(self, model_output: AgentOutput):

		message = f"{model_output.evaluation_previous_goal}, {model_output.memory}, {model_output.next_goal}, "
		choice = model_output.get_choice()
		choice_dump = model_output.action.choice.model_dump()

		if choice == 'agent':
			agent_name, agent_message = get_first_key_param(choice_dump)
			message += f"For this i will call agent '{agent_name}' and assign it task '{agent_message}'\n"
		elif choice == 'tool':
			message += f"for this i will call tool : {choice_dump}"
		else:
			raise ValueError('Unknow choice')
		
		return message

	def add_choice(self, model_output: AgentOutput) -> None:
		choice = model_output.get_choice()
		choice_json = model_output.action.choice.model_dump()

		if choice == 'agent':
			agent_name, message = get_first_key_param(choice_json)
			tool_calls = [
				{
					'name': agent_name,
					'args': {'message':message},
					'id': agent_name,
					'type': 'tool_call',
				}
			]

			msg = AIMessage(
				content=self.format_agentoutput(model_output),
				tool_calls=tool_calls
			)
		elif choice == 'tool':
			key_name, param = get_first_key_param(choice_json)

			tool_calls = [
				{
					'name': key_name,
					'args': param,
					'id': key_name,
					'type': 'tool_call',
				}
			]

			msg = AIMessage(
				content=self.format_agentoutput(model_output),
				tool_calls=tool_calls
			)
		else:
			raise ValueError("Unknow choice")
		
		self._add_message_with_tokens(msg)

	def add_response(self, action: AgentOutput, result: str) -> None:
		key_name, param = get_first_key_param(action.action.choice.model_dump())
		tool_message = ToolMessage(
			content=result,
			tool_call_id=key_name,
		)
		self._add_message_with_tokens(tool_message)

	def get_messages(self) -> List[BaseMessage]:
		"""Get current message list, potentially trimmed to max tokens"""

		msg = [m.message for m in self.history.messages]
		# debug which messages are in history with token count # log

		total_input_tokens = 0
		logger.info(f'Messages in history: {len(self.history.messages)}:')
		for m in self.history.messages:
			total_input_tokens += m.metadata.input_tokens

		logger.info(f'Total input tokens: {total_input_tokens}')

		return msg
	
	def _add_message_with_tokens(self, message: BaseMessage) -> None:
		"""Add message with token count metadata"""
		token_count = self._count_tokens(message)
		metadata = MessageMetadata(input_tokens=token_count)
		self.history.add_message(message, metadata)

	def _count_tokens(self, message: BaseMessage) -> int:
		"""Count tokens in a message using the model's tokenizer"""
		tokens = 0
		if isinstance(message.content, list):
			for item in message.content:
				if 'image_url' in item:
					tokens += self.IMG_TOKENS
				elif isinstance(item, dict) and 'text' in item:
					tokens += self._count_text_tokens(item['text'])
		else:
			msg = message.content
			if hasattr(message, 'tool_calls'):
				msg += str(message.tool_calls)  # type: ignore
			tokens += self._count_text_tokens(msg)
		return tokens

	def _count_text_tokens(self, text: str) -> int:
		"""Count tokens in a text string"""
		tokens = len(text) // self.estimated_characters_per_token  # Rough estimate if no tokenizer available
		return tokens

	def cut_messages(self):
		"""Get current message list, potentially trimmed to max tokens"""
		diff = self.history.total_tokens - self.max_input_tokens
		if diff <= 0:
			return None

		msg = self.history.messages[-1]

		# if list with image remove image
		if isinstance(msg.message.content, list):
			text = ''
			for item in msg.message.content:
				if 'image_url' in item:
					msg.message.content.remove(item)
					diff -= self.IMG_TOKENS
					msg.metadata.input_tokens -= self.IMG_TOKENS
					self.history.total_tokens -= self.IMG_TOKENS
					logger.debug(
						f'Removed image with {self.IMG_TOKENS} tokens - total tokens now: {self.history.total_tokens}/{self.max_input_tokens}'
					)
				elif 'text' in item and isinstance(item, dict):
					text += item['text']
			msg.message.content = text
			self.history.messages[-1] = msg

		if diff <= 0:
			return None

		# if still over, remove text from state message proportionally to the number of tokens needed with buffer
		# Calculate the proportion of content to remove
		proportion_to_remove = diff / msg.metadata.input_tokens
		if proportion_to_remove > 0.99:
			raise ValueError(
				f'Max token limit reached - history is too long - reduce the system prompt or task. '
				f'proportion_to_remove: {proportion_to_remove}'
			)
		logger.debug(
			f'Removing {proportion_to_remove * 100:.2f}% of the last message  {proportion_to_remove * msg.metadata.input_tokens:.2f} / {msg.metadata.input_tokens:.2f} tokens)'
		)

		content = msg.message.content
		characters_to_remove = int(len(content) * proportion_to_remove)
		content = content[:-characters_to_remove]

		# remove tokens and old long message
		self.history.remove_message(index=-1)

		# new message with updated content
		msg = HumanMessage(content=content)
		self._add_message_with_tokens(msg)

		last_msg = self.history.messages[-1]

		logger.debug(
			f'Added message with {last_msg.metadata.input_tokens} tokens - total tokens now: {self.history.total_tokens}/{self.max_input_tokens} - total messages: {len(self.history.messages)}'
		)

	def convert_messages_for_non_function_calling_models(self, input_messages: list[BaseMessage]) -> list[BaseMessage]:
		"""Convert messages for non-function-calling models"""
		output_messages = []
		for message in input_messages:
			if isinstance(message, HumanMessage):
				output_messages.append(message)
			elif isinstance(message, SystemMessage):
				output_messages.append(message)
			elif isinstance(message, ToolMessage):
				output_messages.append(HumanMessage(content=message.content))
			elif isinstance(message, AIMessage):
				# check if tool_calls is a valid JSON object
				if message.tool_calls:
					tool_calls = json.dumps(message.tool_calls)
					output_messages.append(AIMessage(content=tool_calls))
				else:
					output_messages.append(message)
			else:
				raise ValueError(f'Unknown message type: {type(message)}')
		return output_messages

	def merge_successive_human_messages(self, messages: list[BaseMessage]) -> list[BaseMessage]:
		"""Some models like deepseek-reasoner dont allow multiple human messages in a row. This function merges them into one."""
		merged_messages:list[BaseMessage] = []
		streak = 0
		for message in messages:
			if isinstance(message, HumanMessage):
				streak += 1
				if streak > 1:
					merged_messages[-1].content += message.content #type:ignore
				else:
					merged_messages.append(message)
			else:
				merged_messages.append(message)
				streak = 0
		return merged_messages

	def extract_json_from_model_output(self, content: str) -> dict:
		"""Extract JSON from model output, handling both plain JSON and code-block-wrapped JSON."""
		try:
			# If content is wrapped in code blocks, extract just the JSON part
			if content.startswith('```'):
				# Find the JSON content between code blocks
				content = content.split('```')[1]
				# Remove language identifier if present (e.g., 'json\n')
				if '\n' in content:
					content = content.split('\n', 1)[1]
			# Parse the cleaned content
			return json.loads(content)
		except json.JSONDecodeError as e:
			logger.warning(f'Failed to parse model output: {str(e)}')
			raise ValueError('Could not parse response.')
