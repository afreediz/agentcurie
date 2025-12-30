# MAIN CONTROLLER LOGIC
from .tool import ToolsController, ToolModel
from .agent import AgentsController, AgentCard, BaseAgent, AgentModel
from .agent import SuperVisor
from .views import ChoiceModel, ChoiceResult
from typing import Type, Union
from pydantic import create_model, Field
from langchain_openai.chat_models.base import BaseChatOpenAI

import logging
logger = logging.getLogger(__name__)

class Controller():
    def __init__(self, supervisor:SuperVisor, llm:BaseChatOpenAI) -> None:
        """
            llm: The language model instance.
            supervisor: Supervisor agent which will get registered inside all other child agents.
        """
        self.agents_controller = AgentsController(supervisor=supervisor, llm=llm)
        self.tools_controller = ToolsController()

    def register_agent(self, agent_card:AgentCard, agent_class:Type[BaseAgent]):
        self.agents_controller.register_agent(card=agent_card, agent_class=agent_class)

    def tool(self, description: str, **kwargs):
        return self.tools_controller.tool(description, **kwargs)

    def create_choice_model(self, exclude_agents:list[str] = [], exclude_tools:list[str] = []) -> Type[ChoiceModel]:#type:ignore
        tools_model = self.tools_controller.registry.create_tool_model(exclude_tools)
        agents_model = self.agents_controller.registry.create_agent_model(exclude_agents)

        union_model = Union[(tools_model, agents_model)]

        model:ChoiceModel = create_model(
            'ChoiceModel',
            __base__=ChoiceModel,
            **{
                'choice':(#type:ignore
                    union_model,
                    Field(..., description="Tool or Agent")
                )
            }
        )

        return model#type:ignore

    def get_prompt_description(self) -> str:
        tools_description = self.tools_controller.registry.get_prompt_description()
        agents_description = self.agents_controller.registry.get_prompt_description()

        description = 'Available tools:'
        description += f'\n {tools_description}'
        description += f'\n Available agents'
        description += f'\n {agents_description}'

        return description

    async def execute_agent(self, choice:AgentModel) -> ChoiceResult:
        try:
            res = await self.agents_controller.act(choice)
            return ChoiceResult(content=res.content, is_done=res.is_done, success=res.success, error=res.error)
        except Exception as e:
            raise e
            return ChoiceResult(error=str(e))

    async def execute_tool(self, choice:ToolModel) -> ChoiceResult:
        try:
            logger.info(f'trying to execute tool : {choice}, {type(choice)}')
            res = await self.tools_controller.act(choice)
            return ChoiceResult(content=res.content, is_done=res.is_done, success=res.success, error=res.error)
        except Exception as e:
            logger.info(f"{str(e)}")
            raise e
            return ChoiceResult(error=str(e))

    async def execute_choise(self, choice:ChoiceModel) -> ChoiceResult:
        choosed_type = choice.get_type()

        try:
            if choosed_type == 'tool':
                assert isinstance(choice.choice.root, ToolModel)
                res = await self.tools_controller.act(choice.choice.root)
            elif choosed_type == 'agent':
                assert isinstance(choice.choice.root, AgentModel)
                res = await self.agents_controller.act(choice.choice.root)
            else:
                raise ValueError("Unknow choice, choice should be either a tool or an agent")
        
            return ChoiceResult(content=res.content, is_done=res.is_done, success=res.success, error=res.error)

        except Exception as e:
            return ChoiceResult(error=str(e))