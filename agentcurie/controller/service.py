# MAIN CONTROLLER LOGIC
from .tool import ToolsController, ToolModel
from .agent import AgentsController, AgentCard, BaseAgent, AgentModel
from .agent import SuperVisor
from .views import ChoiceModel, ChoiceResult, ToolsAction, AgentAction
from typing import Type, Union
from pydantic import create_model, Field
from langchain_core.language_models.chat_models import BaseChatModel

import logging
logger = logging.getLogger(__name__)

class Controller():
    def __init__(self, supervisor:SuperVisor, llm:BaseChatModel) -> None:
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

        ToolsActionModel = create_model(
            'ToolsAction',
            __base__=ToolsAction,
            tools=(list[tools_model], Field(..., description='One or more tools to execute in sequence')),  # type: ignore
        )

        AgentActionModel = create_model(
            'AgentAction',
            __base__=AgentAction,
            agent=(agents_model, Field(..., description='Exactly one agent to delegate the task to')),
        )

        model: ChoiceModel = create_model(
            'ChoiceModel',
            __base__=ChoiceModel,
            choice=(  # type: ignore
                Union[ToolsActionModel, AgentActionModel],
                Field(..., description='Tool(s) to run or a single Agent to delegate to'),
            ),
        )

        return model  # type: ignore

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
            logger.debug(f'trying to execute tool : {choice}, {type(choice)}')
            res = await self.tools_controller.act(choice)
            return ChoiceResult(content=res.content, is_done=res.is_done, success=res.success, error=res.error) #type:ignore
        except Exception as e:
            logger.info(f"{str(e)}")
            raise e
            return ChoiceResult(error=str(e))

    async def execute_choise(self, choice: ChoiceModel) -> ChoiceResult:
        try:
            if isinstance(choice.choice, ToolsAction):
                results = []
                for tool_model in choice.choice.tools:
                    res = await self.tools_controller.act(tool_model)
                    results.append(res.content or '')
                    if res.is_done:
                        return ChoiceResult(content=res.content, is_done=True, success=res.success) #type:ignore
                return ChoiceResult(content='\n'.join(results))
            elif isinstance(choice.choice, AgentAction):
                res = await self.agents_controller.act(choice.choice.agent)
                return ChoiceResult(content=res.content, is_done=res.is_done, success=res.success, error=res.error)
            else:
                raise ValueError("Unknown choice — must be ToolsAction or AgentAction")
        except Exception as e:
            return ChoiceResult(error=str(e))