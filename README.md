# AgentCurie

AgentCurie is a **supervisor agentic framework** designed to control and coordinate agents built with multiple frameworks.

The name is inspired by **Maria Salomea Skłodowska-Curie**, the only person to have received Nobel Prizes in two different scientific fields. Similarly, AgentCurie is capable of supervising and orchestrating agents originating from different ecosystems.

---

## 🎯 Purpose

AgentCurie acts as a **master supervisor** that:

* Manages multiple agents, even if they are built using different frameworks
* Coordinates tools, agents, and execution flow
* Provides a structured, extensible foundation for agentic systems

---

## ⚡ Quick Start

This section shows how to get up and running with **AgentCurie** by creating a simple agent and supervising it.

### 1️⃣ Define Your Agent

Create a custom agent by extending `BaseAgent`. Your agent implements a `process` method, which the supervisor will invoke.

```python
from agentcurie import BaseAgent

class CreativeAgent(BaseAgent):
    def __init__(self):
        super().__init__()

        # Tool to communicate back with the supervisor
        query_tool = tool()(self.query_supervisor)

        # Your underlying LLM-powered agent
        self.agent = create_agent(
            model=llm,
            tools=[calculator, write_poem, query_tool],
            system_prompt=(
                "You are a creative assistant that can perform calculations "
                "and write poems. Be helpful and creative!"
            )
        )

    # This method is called by the supervisor
    async def process(self, message: str) -> str:
        result = await self.agent.ainvoke({
            "messages": [{"role": "user", "content": message}]
        })

        return result["messages"][-1].content
```

---

### 2️⃣ Describe the Agent with an AgentCard

An `AgentCard` defines the agent’s identity, skills, and lifecycle behavior.

```python
from agentcurie import AgentCard

my_agent_card = AgentCard(
    name="creative_agent",
    description="Can write poems and perform calculations",
    skills=[
        "write poems",
        "does calculations like add, subtract, multiply, divide"
    ],
    persistent=True  # Keep the agent alive across invocations
)
```

---

### 3️⃣ Create and Configure the Supervisor

The `SupervisorAgent` manages agents and tools and decides how to route tasks.

```python
from agentcurie import SupervisorAgent

supervisor = SupervisorAgent(llm=llm)

supervisor.register_agent(
    agent_card=my_agent_card,
    agent_class=CreativeAgent
)
```

---

### 4️⃣ Register Supervisor-Level Tools

You can attach tools directly to the supervisor. These tools are available during task execution.

```python
@supervisor.register_tool("Use to get weather details of any place")
def get_weather(city: str):
    """Get current weather for a city (mock function)"""
    weather_data = {
        "New York": "Sunny, 72°F",
        "London": "Cloudy, 15°C",
        "Tokyo": "Rainy, 18°C",
        "Paris": "Partly cloudy, 20°C"
    }
    return weather_data.get(city, f"Weather data not available for {city}")
```

---

### 5️⃣ Run the Supervisor

Call `solve()` on the supervisor with a natural-language task. The supervisor will:

* Decide which tools to use
* Invoke the appropriate agent(s)
* Coordinate the execution flow

```python
import asyncio

async def main():
    result = await supervisor.solve(
        "Check weather details of London, write a poem of the current weather"
    )
    return result

asyncio.run(main())
```

---

### ✅ What Happens Internally?

1. The supervisor analyzes the task
2. The weather tool is invoked
3. The creative agent is selected and initialized
4. The agent processes the request and returns the final output

This demonstrates how **AgentCurie acts as a central brain**, orchestrating agents and tools seamlessly.

---

## 📁 Code Structure

AgentCurie follows a **feature-based modular architecture**, promoting:

* Clear separation of concerns
* Scalability for large systems
* High testability and maintainability

```
feature_1/
├── service.py       # Core business logic and orchestration
├── views.py         # Pydantic models (request/response schemas)
├── model.py         # Database or domain models
├── test.py          # Feature-specific tests
├── example/         # Usage examples and demos
└── sub_feature/     # Optional nested features

feature_2/
├── ...
```

Each feature is **self-contained** and can evolve independently.

---

## 🚀 Highlights

1. **Multi-agent orchestration**
  Seamlessly integrate and control agents from different frameworks.

2. **Agent Queries**
  Child agents can ask query/service back to supervisor which will temperorly hold the child agent till supervisor respond directly or by coordinating with other agents.  

3. **Dynamic Initialisation and Persistance**
  Child agents are initialised only when necessary and can be set to persist until completion.

4. **Testable by design**
  Every feature includes its own test suite.

5. **Agentic-compatible**
  Designed to work naturally with modern LLM tools, planners, and controllers.

5. **Structured automation**
  Clean separation between data models, business logic, and views.

---

## 📂 Examples

Each feature contains an `examples/` directory that demonstrates:

* How to interact with the feature’s services
* How agents are executed and coordinated
* Typical usage patterns for the framework

These examples are intended as both **learning resources** and **quick-start references**.

---

## 🧭 Important Code Guide

Key directories and their responsibilities:

* `supervisor/`
  Contains the implementation of the **master supervisor agent**, responsible for coordinating agents and tools across frameworks.

* `controller/tool/`
  Manages tool registration, execution, and lifecycle.

* `controller/agent/`
  Handles agent management and coordination logic.

* `mcp_client/`
  Converts MCP-compatible definitions into tools usable by the tool controller.

---

## 🔮 Vision

AgentCurie is designed as a **framework-agnostic control layer** for the future of agentic systems—where multiple agents, tools, and reasoning engines collaborate under a single, well-structured supervisor.
