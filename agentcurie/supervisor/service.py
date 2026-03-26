from agentcurie.controller import Controller, AgentOutput, ChoiceModel, BaseAgent, AgentCard, SuperVisor, ToolsAction, AgentAction, ToolResult
from .message_manager import MessageManager
from .prompts import SystemPrompt
from .utils import get_first_key_param
from .views import AgentResult, AgentContext, AgentStatus, FuncHook, AgentHook
from langchain_core.language_models.chat_models import BaseChatModel
from typing import Type, TypeVar, Optional
from pydantic import BaseModel
from typing import Dict

import os
import asyncio

import logging
logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

class SupervisorAgent(SuperVisor):
    def __init__(self, llm:BaseChatModel, func_hooks:list[FuncHook] = [], agent_hooks:list[AgentHook] = [], extended_system_prompt:str|None = None):
        self.llm = llm
        self.controller = Controller(supervisor=self, llm=llm)

        # func hooks - function which gets executed between steps also has access to whole supervisor
        self.func_hooks_before = [hook for hook in func_hooks if hook.order == 'before']
        self.func_hooks_after = [hook for hook in func_hooks if hook.order == 'after']

        # agent hooks - function which gets executed between steps also has access to whole supervisor and mentioned agents
        self.agent_hooks_before = [hook for hook in agent_hooks if hook.order == 'before']
        self.agent_hooks_after = [hook for hook in agent_hooks if hook.order == 'after']

        # contexts to store child agent tasks
        self.agent_tasks: Dict[str, AgentContext] = {}
        self.pending_queries = {}

        # events for agent status signalling (avoids asyncio.sleep polling)
        self._agent_events: Dict[str, asyncio.Event] = {}   # fires on COMPLETED / ERROR / WAITING_FOR_QUERY
        self._query_events: Dict[str, asyncio.Event] = {}   # fires on QUERY_PROCESSED

        # background tool completion queue: list of (tool_name, result_str)
        self.background_tool_results: list[tuple[str, str]] = []

        self.extended_system_prompt = extended_system_prompt

        # ── Built-in: wait_for_agent ──────────────────────────────────────────
        # Registered after all other state is initialised so the closure can
        # safely reference self.
        supervisor_ref = self

        @self.controller.tool(
            "Wait for a background agent to complete and retrieve its result. "
            "Use this when you previously started an agent with run_in_background=True "
            "and now need its result to continue the task.",
        )
        async def wait_for_agent(agent_name: str) -> ToolResult:
            if agent_name not in supervisor_ref.agent_tasks:
                return ToolResult(
                    content=f"No running agent named '{agent_name}' found.",
                    success=False,
                )
            ctx = supervisor_ref.agent_tasks[agent_name]
            if not ctx.is_background:
                return ToolResult(
                    content=f"Agent '{agent_name}' is not running in background.",
                    success=False,
                )
            # Already done by the time we're called
            if ctx.status == AgentStatus.COMPLETED:
                ctx.is_background = False
                return ToolResult(content=str(ctx.result))
            if ctx.status == AgentStatus.ERROR:
                ctx.is_background = False
                return ToolResult(content=f"Agent '{agent_name}' failed: {ctx.error}", success=False)
            # Block until the background task fires its event
            await supervisor_ref._agent_events[agent_name].wait()
            ctx = supervisor_ref.agent_tasks[agent_name]
            ctx.is_background = False
            if ctx.status == AgentStatus.ERROR:
                return ToolResult(content=f"Agent '{agent_name}' failed: {ctx.error}", success=False)
            return ToolResult(content=str(ctx.result) if ctx.result else f"Agent '{agent_name}' completed with no result.")

        # ─────────────────────────────────────────────────────────────────────

        # updated at runtime
        # Registered tools and agents schema for model guiding
        self.AgentOutputModel:Type[ChoiceModel] = self.controller.create_choice_model()
        self.AgentOutput: Type[AgentOutput] = AgentOutput.type_with_custom_tools_and_agents(self.AgentOutputModel)

        # updated at runtime
        # without done model schemas to avoid task ending while another agent is waiting for response
        self.AgentOutputModelWithoutDone:Type[ChoiceModel] = self.controller.create_choice_model(exclude_tools=['done'])
        self.AgentOutputWithoutDone: Type[AgentOutput] = AgentOutput.type_with_custom_tools_and_agents(self.AgentOutputModelWithoutDone)

        self.message_manager = MessageManager()

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
            system_prompt = SystemPrompt(extended_system_prompt=self.extended_system_prompt)
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

        # prepare a fresh query event for this round-trip
        if agent_name not in self._query_events:
            self._query_events[agent_name] = asyncio.Event()
        query_event = self._query_events[agent_name]
        query_event.clear()

        # update agent_task status so main logic can continue and process the query
        self.pending_queries[agent_name] = message
        self.agent_tasks[agent_name].query = message
        self.agent_tasks[agent_name].status = AgentStatus.WAITING_FOR_QUERY

        # wake the supervisor so it can handle the query
        self._agent_events[agent_name].set()

        # wait until the supervisor has posted an answer
        await query_event.wait()

        result = self.agent_tasks[agent_name].result
        assert isinstance(result, str)

        # reset state
        self.agent_tasks[agent_name].result = None
        self.agent_tasks[agent_name].query = None

        del self.pending_queries[agent_name]

        return result

    async def run_child_agent_with_context(self, agent_name, choice) -> None:
        "Assign and execute tasks to child agent with context status"
        try:
            result = await self.controller.execute_agent(choice)
            self.agent_tasks[agent_name].result = result.content
            self.agent_tasks[agent_name].status = AgentStatus.COMPLETED
            self._agent_events[agent_name].set()
        except Exception as e:
            logger.info(f"{agent_name} produced error {str(e)}")
            self.agent_tasks[agent_name].status = AgentStatus.ERROR
            self.agent_tasks[agent_name].error = str(e)
            self._agent_events[agent_name].set()
            raise e

    async def run_background_tool(self, tool_name: str, tool_model) -> None:
        """Execute a tool as a fire-and-forget background task.

        On completion the result is appended to self.background_tool_results so
        the supervisor picks it up at the start of the next loop iteration.
        """
        try:
            result = await self.controller.execute_tool(tool_model)
            content = result.content if result.content is not None else "Tool completed with no output."
            self.background_tool_results.append((tool_name, str(content)))
            logger.info(f"\033[36m ✅ [bg] {tool_name} completed\033[0m")
        except Exception as e:
            self.background_tool_results.append((tool_name, f"Error: {str(e)}"))
            logger.info(f"\033[31m ❌ [bg] {tool_name} failed: {str(e)}\033[0m")

    def _check_background_completions(self) -> list[tuple[str, str, Optional[str], Optional[str]]]:
        """Scan agent_tasks for newly completed / querying background agents.

        Returns a list of (agent_name, event_type, result, extra) where
        event_type is one of 'completed', 'error', or 'query'.
        Each returned entry is marked so it won't be reported twice.
        """
        events: list[tuple[str, str, Optional[str], Optional[str]]] = []
        for agent_name, ctx in list(self.agent_tasks.items()):
            if not ctx.is_background:
                continue
            if ctx.status == AgentStatus.COMPLETED:
                events.append((agent_name, 'completed', ctx.result, None))
                ctx.is_background = False  # processed
            elif ctx.status == AgentStatus.ERROR:
                events.append((agent_name, 'error', None, ctx.error))
                ctx.is_background = False  # processed
            elif ctx.status == AgentStatus.WAITING_FOR_QUERY:
                # Background agent paused to ask a query; intercept once
                events.append((agent_name, 'query', None, ctx.query))
                ctx.status = AgentStatus.QUERY_INTERCEPTED
        return events

    async def solve(self, task: str):
        "Main entry point"
        try:
            # prepare agent
            await self.prepare_agent(task)

            step = 1
            while True:
                msg_count = len(self.message_manager.history.messages)
                token_count = self.message_manager.history.total_tokens
                logger.info(f"\033[1m\033[34m{'─' * 6} Step {step} │ msgs:{msg_count} │ tokens:{token_count} {'─' * 6}\033[0m")

                # ── Inject background-completion notifications ─────────────────
                for agent_name, event_type, result, extra in self._check_background_completions():
                    if event_type == 'completed':
                        msg = f"[BACKGROUND UPDATE] Agent '{agent_name}' has completed. Result: {result}"
                    elif event_type == 'error':
                        msg = f"[BACKGROUND UPDATE] Agent '{agent_name}' failed with error: {extra}"
                    else:  # 'query'
                        msg = (
                            f"[BACKGROUND UPDATE] Background agent '{agent_name}' is paused "
                            f"and has a query: {extra}"
                        )
                    logger.info(f"\033[35m {msg} \033[0m")
                    self.message_manager.add_human_message(msg)

                while self.background_tool_results:
                    tool_name, tool_result = self.background_tool_results.pop(0)
                    msg = f"[BACKGROUND UPDATE] Tool '{tool_name}' has completed. Result: {tool_result}"
                    logger.info(f"\033[35m {msg} \033[0m")
                    self.message_manager.add_human_message(msg)
                # ─────────────────────────────────────────────────────────────

                messages = self.message_manager.get_messages()

                # get response model, also avoiding task completion when there is pending queries
                response_model = self.AgentOutput if not self.pending_queries else self.AgentOutputWithoutDone
                logger.debug(f"Using response model {response_model}")

                next_action:AgentOutput = await self.get_structured_response(messages, response_model)

                self.message_manager.add_choice(next_action)
                logger.info(f"\033[32m 🔍 {next_action.evaluation_previous_goal}\033[0m")
                logger.info(f"\033[2m 📝 {next_action.memory}\033[0m")
                logger.info(f"\033[33m 💡 {next_action.next_goal}\033[0m")

                choice = next_action.action.choice  # ToolsAction | AgentAction

                if isinstance(choice, AgentAction):
                    agent_model = choice.agent
                    agent_name, agent_task = get_first_key_param(agent_model.model_dump())

                    # Determine whether this invocation should run in background
                    run_in_background = getattr(choice, 'run_in_background', False)
                    registered = self.controller.agents_controller.registry.registry.agents.get(agent_name)
                    bg_capable = bool(registered and registered.card.background_runnable)
                    should_bg = run_in_background and bg_capable

                    if agent_name in self.pending_queries:
                        # ── Respond to a pending query (may be from a background agent) ──
                        pending_query = self.pending_queries[agent_name]

                        # Fresh event so a future wait unblocks correctly
                        self._agent_events[agent_name] = asyncio.Event()

                        self.agent_tasks[agent_name].result = agent_task
                        self.agent_tasks[agent_name].status = AgentStatus.QUERY_PROCESSED
                        self._query_events[agent_name].set()
                        logger.info(f"\033[32m 📨 {agent_name} ← '{agent_task}' (answering: '{pending_query}')\033[0m")

                        if self.agent_tasks[agent_name].is_background:
                            # Background agent: don't wait — it will continue on its own
                            message = (
                                f"Answer sent to background agent '{agent_name}'. "
                                f"You will be notified when it completes."
                            )
                            self.message_manager.add_response(next_action, message)
                        else:
                            # Foreground agent: wait for it to finish or ask next query
                            await self._agent_events[agent_name].wait()

                            res, status = self.agent_tasks[agent_name].result, self.agent_tasks[agent_name].status

                            if status in (AgentStatus.COMPLETED, AgentStatus.ERROR):
                                message = f"Result from agent - {agent_name}: {res}"
                                logger.info(f"\033[32m ✅ {agent_name}: {res}\033[0m")
                                self.message_manager.add_response(next_action, message)
                                for hook in self.agent_hooks_after:
                                    await hook.func(self.controller.agents_controller.registry.get_agent_instance(hook.agent_name), self)

                            elif status == AgentStatus.WAITING_FOR_QUERY:
                                self.agent_tasks[agent_name].status = AgentStatus.QUERY_INTERCEPTED
                                message = f"While processing request, agent {agent_name} has a query: {self.agent_tasks[agent_name].query}"
                                logger.info(f"\033[33m 💬 {agent_name} asks: {self.agent_tasks[agent_name].query}\033[0m")
                                self.message_manager.add_response(next_action, message)

                    else:
                        # ── New agent invocation ───────────────────────────────────────
                        self.agent_tasks[agent_name] = AgentContext(
                            message=agent_task,
                            agent_name=agent_name,
                            is_background=should_bg,
                        )
                        self._agent_events[agent_name] = asyncio.Event()

                        for hook in self.agent_hooks_before:
                            await hook.func(self.controller.agents_controller.registry.get_agent_instance(hook.agent_name), self)

                        bg_tag = " [bg]" if should_bg else ""
                        logger.info(f"\033[35m 🤖 {agent_name}{bg_tag} ← '{agent_task}'\033[0m")
                        asyncio.create_task(self.run_child_agent_with_context(agent_name, agent_model))

                        if should_bg:
                            # Fire-and-forget: let the supervisor continue
                            self.agent_tasks[agent_name].status = AgentStatus.RUNNING_IN_BACKGROUND
                            message = (
                                f"Agent '{agent_name}' has been started in background. "
                                f"You will be notified when it completes. "
                                f"Use the wait_for_agent tool if you need to block on its result."
                            )
                            logger.info(f"\033[2m    ↳ started in background\033[0m")
                            self.message_manager.add_response(next_action, message)
                        else:
                            # Wait for the agent to reach COMPLETED / ERROR / WAITING_FOR_QUERY
                            await self._agent_events[agent_name].wait()

                            res, status = self.agent_tasks[agent_name].result, self.agent_tasks[agent_name].status

                            if status in (AgentStatus.COMPLETED, AgentStatus.ERROR):
                                message = f"Result from agent - {agent_name}: {res}"
                                logger.info(f"\033[32m ✅ {agent_name}: {res}\033[0m")
                                self.message_manager.add_response(next_action, message)
                                for hook in self.agent_hooks_after:
                                    await hook.func(self.controller.agents_controller.registry.get_agent_instance(hook.agent_name), self)

                            elif status == AgentStatus.WAITING_FOR_QUERY:
                                self.agent_tasks[agent_name].status = AgentStatus.QUERY_INTERCEPTED
                                message = f"While processing request, agent {agent_name} has a query: {self.agent_tasks[agent_name].query}"
                                logger.info(f"\033[33m 💬 {agent_name} asks: {self.agent_tasks[agent_name].query}\033[0m")
                                self.message_manager.add_response(next_action, message)

                elif isinstance(choice, ToolsAction):
                    for tool_model in choice.tools:
                        dump = tool_model.model_dump()
                        run_tool_in_bg = dump.get('run_in_background', False)
                        tool_name, params = get_first_key_param(dump)

                        # Validate background capability against registry
                        registered_tool = self.controller.tools_controller.registry.registry.tools.get(tool_name)
                        tool_bg_capable = bool(registered_tool and registered_tool.background_runnable)
                        should_tool_bg = run_tool_in_bg and tool_bg_capable

                        bg_tag = " [bg]" if should_tool_bg else ""
                        logger.info(f"\033[36m 🔧 {tool_name}{bg_tag} ← {params}\033[0m")

                        for hook in self.func_hooks_before:
                            await hook.func(self)

                        if should_tool_bg:
                            asyncio.create_task(self.run_background_tool(tool_name, tool_model))
                            bg_msg = (
                                f"Tool '{tool_name}' started in background. "
                                f"You will be notified when it completes."
                            )
                            logger.info(f"\033[2m    ↳ started in background\033[0m")
                            self.message_manager.add_tool_message(bg_msg, tool_name)
                            result = None  # no synchronous result
                        else:
                            result = await self.controller.execute_tool(tool_model)
                            assert isinstance(result.content, str)
                            self.message_manager.add_tool_message(result.content, tool_name)

                        for hook in self.func_hooks_after:
                            await hook.func(self)

                        logger.info(f"\033[32m    ↳ {result.content if result else '(running in background)'}\033[0m")

                        if result and result.is_done:
                            break

                    if result and result.is_done:
                        logger.info(f"\033[1m\033[32m ✅ DONE — {result.content} │ total tokens: {self.message_manager.history.total_tokens}\033[0m")
                        break

                step += 1

            # post processing
            logger.info(f"Task completed : {result.content}")
            return AgentResult(content=result.content, messages=self.message_manager.get_messages())

        except Exception as e:
            raise e

if __name__ == "__main__":
    pass
