from agentcurie.controller import Controller, AgentOutput, ChoiceModel, BaseAgent, AgentCard, SuperVisor
from agentcurie.controller.agent import AgentModel
from agentcurie.controller.tool import ToolModel
from .message_manager import MessageManager
from .prompts import SystemPrompt
from .utils import get_first_key_param
from .views import AgentResult, AgentContext, AgentStatus, FuncHook, AgentHook
from langchain_openai import AzureChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI
from typing import Type, TypeVar
from pydantic import BaseModel
from typing import Dict, Any
from uuid import uuid4

import os
import asyncio

import logging
logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

class SupervisorAgent(SuperVisor):
    def __init__(self, llm:BaseChatOpenAI, func_hooks:list[FuncHook] = [], agent_hooks:list[AgentHook] = []):
        self.llm = llm
        self.controller = Controller(supervisor=self, llm=llm)
        
        # func hooks - function which gets executed between steps also has access to whole supervisor
        self.func_hooks_before = [hook for hook in func_hooks if hook.order == 'before']
        self.func_hooks_after = [hook for hook in func_hooks if hook.order == 'after']

        # agent hooks - function which gets executed between steps also has access to whole supervisor and mentioned agents
        self.agent_hooks_before = [hook for hook in agent_hooks if hook.order == 'before']
        self.agent_hooks_after = [hook for hook in agent_hooks if hook.order == 'after']

        # updated at runtime
        # Registered tools and agents schema ofr model guiding
        self.AgentOutputModel:Type[ChoiceModel] = self.controller.create_choice_model()
        self.AgentOutput: Type[AgentOutput] = AgentOutput.type_with_custom_tools_and_agents(self.AgentOutputModel)

        # updated at runtime
        # without done model schemas to avoid task ending while another agent is waiting for response
        self.AgentOutputModelWithoutDone:Type[ChoiceModel] = self.controller.create_choice_model(exclude_tools=['done'])
        self.AgentOutputWithoutDone: Type[AgentOutput] = AgentOutput.type_with_custom_tools_and_agents(self.AgentOutputModelWithoutDone)

        self.message_manager = MessageManager()

        # contexts to store child agent tasks
        self.agent_tasks: Dict[str, AgentContext] = {}
        self.pending_queries = {}

    def register_agent(self, agent_card:AgentCard, agent_class:Type[BaseAgent]):
        self.controller.register_agent(agent_card=agent_card, agent_class=agent_class)

    def register_tool(self, description: str, **kwargs):
        return self.controller.tool(description, **kwargs)

    async def get_structured_response(self, messages, response_model:  Type[T]) -> T:
        """Get structured response using LangChain"""
        try:
            response = await self.llm.with_structured_output(response_model).ainvoke(messages)
            return response #type:ignore
        except Exception as e:
            raise e

    async def prepare_agent(self, task) -> None:
        # update system prompt
        logger.info("\033[33m ---- CONTEXT INITIALIZING STARTED ---- \033[0m")
        if len(self.message_manager.history.messages) == 0:
            logger.info('updating system prompt')
            system_prompt = SystemPrompt()
            system_message = system_prompt.get_system_message(self.controller.get_prompt_description())
            self.message_manager._add_message_with_tokens(system_message)
            # update model schemas
            logger.info('creating agent model')
            AgentOutputModel = self.controller.create_choice_model()
            self.AgentOutputModel = AgentOutputModel
            self.AgentOutput = AgentOutput.type_with_custom_tools_and_agents(AgentOutputModel) #type:ignore

            AgentOutputModelWithoutDone = self.controller.create_choice_model(exclude_tools=['done'])
            self.AgentOutputModelWithoutDone = AgentOutputModelWithoutDone
            self.AgentOutputWithoutDone = AgentOutput.type_with_custom_tools_and_agents(AgentOutputModelWithoutDone)

        # update task instructions
        logger.info('updating task instruction')
        self.message_manager.add_new_task(task)
        logger.info("\033[33m ---- CONTEXT INITIALIZING COMPLETED ---- \033[0m")

    async def _solve_query(self, message, agent_name) -> str:
        """Resolve query from child agents"""

        if agent_name not in self.agent_tasks:
            raise RuntimeError("Query from a unknow agent task")
        
        # update agent_task status so main logic can continue and process the query
        self.pending_queries[agent_name] = message
        self.agent_tasks[agent_name].query = message
        self.agent_tasks[agent_name].status = AgentStatus.WAITING_FOR_QUERY

        while self.agent_tasks[agent_name].status != AgentStatus.QUERY_PROCESSED:
            await asyncio.sleep(1)

        result = self.agent_tasks[agent_name].result
        assert isinstance(result, str)

        # reset state
        # del self.agent_tasks[agent_name]
        self.agent_tasks[agent_name].result = None
        self.agent_tasks[agent_name].query = None

        del self.pending_queries[agent_name]

        return result

    async def run_child_agent_with_context(self, agent_name, choice:AgentModel) -> None:
        "Assign and execute tasks to child agent with context status"
        try:
            result = await self.controller.execute_agent(choice)
            self.agent_tasks[agent_name].result = result.content
            self.agent_tasks[agent_name].status = AgentStatus.COMPLETED
        except Exception as e:
            logger.info(f"{agent_name} produced error {str(e)}")
            raise e

    async def solve(self, task: str):
        "Main entry point"
        try:
            # prepare agent
            await self.prepare_agent(task)

            step = 1
            while True:
                logger.info(f"Task : {task}")
                logger.info(f"🪜 Step : {step}")
                messages = self.message_manager.get_messages()

                # get response model, also avoiding task completion when there is pending queries
                response_model = self.AgentOutput if not self.pending_queries else self.AgentOutputWithoutDone
                logger.info(f"Using response model {response_model}")
                
                next_action:AgentOutput = await self.get_structured_response(messages, response_model)
                # logger.info(f"Resoponse from model : {next_action.model_dump()}")

                self.message_manager.add_choice(next_action)
                logger.info(f"\033[32m 🔍{next_action.evaluation_previous_goal} \n📝{next_action.memory} \n💡{next_action.next_goal} \033[0m")
                
                try:
                    choice = next_action.action.choice.root
                except Exception as e:
                    choice = next_action.action.choice

                if next_action.get_choice() == 'agent':
                    assert isinstance(choice, AgentModel)

                    agent_name, agent_task = get_first_key_param(choice.model_dump())
                    self.agent_tasks[agent_name] = AgentContext(message=agent_task, agent_name=agent_name)
                    

                    if agent_name in self.pending_queries:
                        # resolve pending queries from child agents
                        pending_query = self.pending_queries[agent_name]
                        self.agent_tasks[agent_name].result = agent_task
                        self.agent_tasks[agent_name].status = AgentStatus.QUERY_PROCESSED
                        logger.info(f"\033[32m Responded to pending query \n{agent_name}:{pending_query}? \nanswer:{agent_task} \033[0m")
                    else:
                        for hook in self.agent_hooks_before:
                            await hook.func(self.controller.agents_controller.registry.get_agent_instance(hook.agent_name), self)

                        logger.info(f"\033[32m Calling agent : {agent_name} - assinging task : '{agent_task}' \033[0m")
                        # run child agent as a background task to enable non-blocking executions
                        asyncio.create_task(self.run_child_agent_with_context(agent_name, choice))
                    
                    # wait for child agent to process and update status
                    while self.agent_tasks[agent_name].status not in (AgentStatus.COMPLETED, AgentStatus.ERROR, AgentStatus.WAITING_FOR_QUERY):
                        await asyncio.sleep(1)

                    res, status = self.agent_tasks[agent_name].result, self.agent_tasks[agent_name].status

                    if status == AgentStatus.COMPLETED or status == AgentStatus.ERROR:
                        message = f"Result from agent - {agent_name}: {res}"
                        logger.info(f"\033[36m {message} \033[0m")

                        self.message_manager.add_response(next_action, message)

                        for hook in self.agent_hooks_after:
                            await hook.func(self.controller.agents_controller.registry.get_agent_instance(hook.agent_name), self)
                    
                    if status == AgentStatus.WAITING_FOR_QUERY:
                        self.agent_tasks[agent_name].status = AgentStatus.QUERY_INTERCEPTED
                        message = f"While processing request, agent {agent_name} has a query: {self.agent_tasks[agent_name].query}"
                        logger.info(f"\033[36m {message} \033[0m")

                        self.message_manager.add_response(next_action, message)

                elif next_action.get_choice() == 'tool':
                    assert isinstance(choice, ToolModel)
                    tool_name, params = get_first_key_param(choice.model_dump())
                    logger.info(f"\033[32m Calling tool : {tool_name} - params: {params} \033[0m")

                    for hook in self.func_hooks_before:
                        await hook.func(self)

                    result = await self.controller.execute_tool(choice)
                    assert isinstance(result.content, str)

                    self.message_manager.add_response(next_action, result.content)

                    for hook in self.func_hooks_after:
                        await hook.func(self)

                    logger.info(f"\033[36m Result from tool - {tool_name} : {result.content} \033[0m")

                    if result.is_done:
                        logger.info(f"\n\nCompleted!, {result.content}")
                        logger.info(f"total tokens: {self.message_manager.history.total_tokens}")
                        break

                step += 1

            # post processing
            logger.info(f"Task completed : {result.content}")
            return AgentResult(content=result.content, messages=self.message_manager.get_messages())

        except Exception as e:
            # return AgentResult(error=str(e))
            raise e

if __name__ == "__main__":
    pass