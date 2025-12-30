# Generative Process Automation (GPA)

GPA (Generative Process Automation) is a modular framework designed to automate agentic tasks using structured, scalable patterns. It draws inspiration from advanced browser-based agentic architectures and employs a format reminiscent of Netflix's Dispatcher design.

## 🧠 Core Inspiration

This framework is inspired by **browser-use agentic architecture**, which has been shown to outperform many other agentic systems in terms of planning, tool usage, and structured task execution.

## 📁 Code Structure

GPA adopts a **feature-based modular design**, encouraging separation of concerns, scalability, and testability.

```
feature_1/
├── service.py       # Core business logic and orchestration
├── views.py         # Pydantic models (request/response schemas)
├── model.py         # Database models
├── test.py          # Module-specific tests
├── example/         # Working examples and usage guides
└── sub_feature/     # Optional nested features
feature_2/
├── ...
```

## 🚀 Highlights

- 🧩 **Feature-oriented**: Each domain feature is isolated and self-contained.
- 🧪 **Testable by design**: Every feature includes its own test suite.
- 🛠️ **Agentic-compatible**: Built for integration with modern LLM tools and controllers.
- 🏗️ **Structured automation**: Clear separation of data models, logic, and views.

## 📂 Examples

Each module has a corresponding `example/` folder showcasing how to interact with its services or simulate agent execution.

---

## Important code guides:
- supervisor/ : contains implemenation of master agent which can control agents and tools from any framework
- controller/tool : contains tools manager
- controller/agent : contains agents manager
- gpa/ : contains an example of using multiple agents and framework with supervisor
- infra/ : contains configurations, db and other service managers
- mcp_client/ : contains logic to convert mcp to tools for tools controller
- recorder/ : contains recorder engine from gpa which can execute browser task with json