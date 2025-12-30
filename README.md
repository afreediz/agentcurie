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

* 🧩 **Multi-agent orchestration**
  Seamlessly integrate and control agents from different frameworks.

* 🧩 **Feature-oriented design**
  Each domain feature is isolated, improving clarity and maintainability.

* 🧪 **Testable by design**
  Every feature includes its own test suite.

* 🛠️ **Agentic-compatible**
  Designed to work naturally with modern LLM tools, planners, and controllers.

* 🏗️ **Structured automation**
  Clean separation between data models, business logic, and views.

---

## 📂 Examples

Each feature contains an `example/` directory that demonstrates:

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
  Handles agent management, routing, and coordination logic.

* `mcp_client/`
  Converts MCP-compatible definitions into tools usable by the tool controller.

* `examples/`
  Provides complete, end-to-end examples demonstrating how to use AgentCurie in real scenarios.

---

## 🔮 Vision

AgentCurie is designed as a **framework-agnostic control layer** for the future of agentic systems—where multiple agents, tools, and reasoning engines collaborate under a single, well-structured supervisor.
