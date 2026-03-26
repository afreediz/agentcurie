# AgentCurie

AgentCurie is a **supervisor agentic framework** designed to control and coordinate agents built with multiple frameworks.

The name is inspired by **Maria Salomea Skłodowska-Curie**, the only person to have received Nobel Prizes in two different scientific fields. Similarly, AgentCurie is capable of supervising and orchestrating agents originating from different ecosystems.

---

## Installation

```bash
pip install agentcurie
```

---

## Quick Start

### 1. Define Your Agent

Create a custom agent by extending `BaseAgent`. Your agent implements a `process` method, which the supervisor will invoke.

```python
from agentcurie import BaseAgent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

class CreativeAgent(BaseAgent):
    def __init__(self):
        super().__init__()

        # Tool to communicate back with the supervisor
        query_tool = tool()(self.query_supervisor)

        self.agent = create_react_agent(
            model=llm,
            tools=[calculator, write_poem, query_tool],
        )

    async def process(self, message: str) -> str:
        result = await self.agent.ainvoke({
            "messages": [{"role": "user", "content": message}]
        })
        return result["messages"][-1].content
```

### 2. Describe the Agent with an AgentCard

An `AgentCard` defines the agent's identity, skills, and lifecycle behavior.

```python
from agentcurie import AgentCard

my_agent_card = AgentCard(
    name="creative_agent",
    description="Can write poems and perform calculations",
    skills=[
        "write poems",
        "does calculations like add, subtract, multiply, divide"
    ],
    persistent=True,           # Keep the agent alive across invocations
    background_runnable=True   # Allow the supervisor to run this agent in background
)
```

### 3. Create and Configure the Supervisor

```python
from agentcurie import SupervisorAgent, FuncHook, AgentHook, BaseAgent

async def intermediate_logger(supervisor: SupervisorAgent):
    last_message = supervisor.message_manager.history.get_last_message()
    print(last_message)

async def child_agent_state_updater(child_agent: BaseAgent, supervisor: SupervisorAgent):
    pass  # custom post-agent logic

supervisor = SupervisorAgent(
    llm=llm,
    func_hooks=[
        FuncHook(order='before', func=intermediate_logger)
    ],
    agent_hooks=[
        AgentHook(order='after', agent_name='creative_agent', func=child_agent_state_updater)
    ]
)
```

### 4. Register Agents and Tools

```python
supervisor.register_agent(
    agent_card=my_agent_card,
    agent_class=CreativeAgent
)

@supervisor.register_tool("Get current weather for a city")
def get_weather(city: str) -> str:
    weather_data = {
        "New York": "Sunny, 72°F",
        "London": "Cloudy, 15°C",
    }
    return weather_data.get(city, f"Weather data not available for {city}")

# Register a background-runnable tool
@supervisor.register_tool("Run a slow background job", background_runnable=True)
async def slow_job(input: str) -> str:
    await asyncio.sleep(5)
    return f"Job done for: {input}"
```

### 5. Run the Supervisor

```python
import asyncio

async def main():
    result = await supervisor.solve(
        "Check weather in London and write a poem about it"
    )
    print(result)

asyncio.run(main())
```

---

## Key Features

### Multi-Agent Orchestration

The supervisor intelligently delegates subtasks to the right child agent and coordinates their results. Child agents can be built with any framework (LangChain, custom, external).

### Agent Queries

If a child agent is stuck or needs additional information, it can call `query_supervisor()` to pause execution and ask the supervisor. The supervisor can answer directly or delegate to another agent.

```python
class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        query_tool = tool()(self.query_supervisor)
        ...
```

### Background Execution

Agents and tools can be marked as `background_runnable`. The supervisor can fire them off without blocking and retrieve results later via `wait_for_agent` / `wait_for_tool`.

**For agents** — set `background_runnable=True` on the `AgentCard`:
```python
AgentCard(name="slow_agent", ..., background_runnable=True)
```

**For tools** — pass `background_runnable=True` to `register_tool`:
```python
@supervisor.register_tool("A slow tool", background_runnable=True)
async def slow_tool(param: str) -> str:
    ...
```

The supervisor automatically receives `[BACKGROUND UPDATE]` notifications when background tasks complete, and can use the built-in `wait_for_agent` or `wait_for_tool` tools to block until a specific result is ready.

### Dynamic Initialization and Persistence

Child agents are only instantiated when first needed. Setting `persistent=True` on an `AgentCard` keeps the agent alive across multiple calls within the same `solve()` session.

### Hook System

Hooks let you inject custom logic before or after any step:

- `FuncHook` — runs before/after each supervisor step (receives the supervisor instance)
- `AgentHook` — runs before/after a specific child agent is invoked (receives the agent and supervisor)

```python
FuncHook(order='before', func=my_func)
AgentHook(order='after', agent_name='my_agent', func=my_agent_func)
```

### MCP Client

AgentCurie includes a built-in MCP (Model Context Protocol) client to convert MCP-compatible tool definitions into supervisor-level tools:

```python
from agentcurie import MCPClient

mcp = MCPClient(...)
tools = await mcp.get_tools()
for t in tools:
    supervisor.register_tool(t.description)(t.func)
```

---

## Architecture

AgentCurie uses a **feature-based modular architecture**:

```
agentcurie/
├── supervisor/          # Master supervisor agent
│   ├── service.py       # SupervisorAgent — main orchestration loop
│   ├── prompts.py       # System prompts
│   ├── views.py         # FuncHook, AgentHook, AgentResult
│   └── message_manager/ # Conversation history & token management
├── controller/          # Tool and agent execution layer
│   ├── service.py       # Controller — dispatches tools and agents
│   ├── tool/            # Tool registration, registry, execution
│   └── agent/           # Agent registration, registry, execution
└── mcp_client/          # MCP protocol integration
```

---

## Running the Example

```bash
# Add your OpenAI API key to examples/.env
echo "OPENAI_KEY=sk-..." > examples/.env

python -m examples.langchain_agents.main
```

---

## Vision

AgentCurie is designed as a **framework-agnostic control layer** for the future of agentic systems — where multiple agents, tools, and reasoning engines collaborate under a single, well-structured supervisor.
