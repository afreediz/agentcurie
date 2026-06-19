# AgentCurie: One Supervisor to Orchestrate Agents From *Any* Framework

### Your agents were built in different ecosystems, speak different dialects, and refuse to cooperate. Here's the conductor they've been missing.

---

## The problem nobody warns you about

You start with one agent. It's beautiful. It answers questions, calls a tool or two, and your demo gets applause.

Then reality arrives.

You need a database agent. Someone on the team already built one — in LangChain. Marketing wants a "creative" agent that writes copy. Your data scientist has a research agent wired up with a completely different stack. And now there's an MCP server your ops team insists you integrate.

Suddenly you're not building *an agent*. You're running an **agent zoo** — and none of the animals speak the same language.

The instinct is to cram everything into one giant agent with fifty tools. We've all tried it. The result is depressingly predictable:

- The model gets **confused** about which of its fifty tools to call.
- The context window **bloats** until reasoning quality nosedives.
- A single slow tool **blocks the entire pipeline**.
- Your orchestration logic hardens into a **brittle, hard-coded chain** that can't adapt when a task takes an unexpected turn.
- And when a sub-task needs clarification, the agent has **no one to ask** — it just hallucinates an answer and marches on.

What you actually want is a *manager*. Someone who understands the goal, knows which specialist to delegate to, can answer their questions, run slow jobs in the background, and only declares victory when the work is genuinely done — regardless of what framework each specialist was built in.

That's **AgentCurie**.

---

## Why "Curie"?

The name is a tribute to **Marie Salomea Skłodowska-Curie** — the only person ever to win Nobel Prizes in *two different sciences* (Physics and Chemistry).

The analogy is deliberate. AgentCurie is built to supervise and coordinate agents that originate from **entirely different ecosystems** — a LangChain agent here, a hand-rolled custom agent there, an MCP-backed tool over there — all under a single, coherent supervisor. One mind, many disciplines.

```bash
pip install agentcurie
```

---

## The 30-second mental model

AgentCurie has exactly three concepts you need to know:

1. **`BaseAgent`** — wrap *any* agent (built with anything) by implementing one async `process(message)` method.
2. **`AgentCard`** — a small manifest describing what an agent does, its skills, and how it behaves.
3. **`SupervisorAgent`** — the orchestrator that reads the cards, plans, delegates, and coordinates.

Here's a complete specialist agent — note that the underlying agent is plain LangChain, but AgentCurie doesn't care:

```python
from agentcurie import BaseAgent
from langchain.agents import create_agent
from langchain_core.tools import tool

class DBAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        # Give the child a tool to talk BACK to the supervisor
        query_tool = tool()(self.query_supervisor)

        self.agent = create_agent(
            model=llm,
            tools=[store_data, get_data, list_all_data, query_tool],
            system_prompt="You manage a database. Store, retrieve, and manage data."
        )

    async def process(self, message: str) -> str:
        result = await self.agent.ainvoke({"messages": [{"role": "user", "content": message}]})
        return result["messages"][-1].content
```

You then hand the supervisor a *card* describing it:

```python
from agentcurie import AgentCard

card = AgentCard(
    name="database_agent",
    description="Performs database operations including storing and querying data",
    skills=["store data", "get data", "list data"],
    persistent=True,            # keep this agent alive across calls
    background_runnable=True,   # supervisor may run it without blocking
)
```

And let the supervisor figure out the rest:

```python
supervisor = SupervisorAgent(llm=llm)
supervisor.register_agent(agent_card=card, agent_class=DBAgent)

result = await supervisor.solve("Store today's sales figures, then write a poem about them")
```

Need to give the supervisor domain knowledge or a house style? Pass `extended_system_prompt` at construction — your text is woven directly into the supervisor's base prompt, so you add rules, tone, or constraints without rewriting the orchestration logic:

```python
supervisor = SupervisorAgent(
    llm=llm,
    extended_system_prompt="You operate in a finance context. Never expose raw PII; "
                           "always confirm a write succeeded before reporting numbers."
)
```

That's it. The supervisor decomposes the task, routes "store the figures" to the database agent and "write a poem" to the creative agent, threads the results together, and returns a final answer. **You never wrote the orchestration logic.** You described capabilities and stated a goal.

Now let's open the hood — because *how* it does this is the interesting part.

---

## Under the hood: the supervisor doesn't "parse," it's *forced* to be valid

The fragile part of every multi-agent system is the moment the LLM picks an action. Free-text reasoning means you're one stray token away from a parsing error or a tool called with garbage arguments.

AgentCurie sidesteps this entirely with **type-guided structured output**. At registration time, it dynamically builds a Pydantic schema for *every* tool and agent you've registered, then unions them into a single decision model:

```python
# Each agent becomes its own typed model...
individual_agent_model = create_model(
    f'{agent_name.title()}AgentModel',
    __base__=AgentModel,
    **{agent_name: (str, Field(..., description=f'message to pass to {agent_name}'))}
)

# ...and the whole decision space becomes a discriminated union:
choice: Union[ToolsAction, AgentAction]
```

Every step, the supervisor asks the model to fill out this exact structure:

```json
{
  "evaluation_previous_goal": "Success — the DB agent confirmed the write.",
  "memory": "Sales figures stored under key 'sales_2026'. Poem still pending.",
  "next_goal": "Delegate poem-writing to the creative agent.",
  "action": { "choice": { "agent": { "creative_agent": "Write a poem about strong sales" } } }
}
```

Because the action is constrained by the schema, the model **can't** invent a non-existent agent or pass malformed arguments — the structured-output layer rejects it and the model retries. The `evaluation / memory / next_goal` triplet isn't decoration either; it forces the supervisor to *reflect before it acts*, which dramatically improves multi-step coherence.

The whole thing runs on LangChain's `BaseChatModel` interface, so AgentCurie is **model-agnostic** — OpenAI, Anthropic, DeepSeek, local models, whatever exposes `with_structured_output`.

---

## Two-way conversation: agents that can *ask for help*

This is my favorite feature, and it's something most orchestration frameworks simply don't have.

In a normal pipeline, delegation is one-way: the manager tells the worker what to do, and the worker either succeeds or fails silently. But real specialists get stuck. They need a missing value, a clarification, a piece of context only someone else has.

In AgentCurie, every child agent can be given a `query_supervisor` tool. When the child calls it, it **pauses mid-execution**, the supervisor is woken up, and the question lands back in the supervisor's reasoning loop. The supervisor can answer directly — or, if it doesn't know, **delegate the question to a *third* agent** to find the answer, then relay it back so the child can pick up exactly where it left off.

So you can express genuinely human workflows. Imagine a customer-support task:

> "Draft a personalized win-back email for customer #4471 offering a discount on the kind of products they actually buy."

The supervisor thinks and assigns the task to **creative agent**, who starts writing — and immediately hits a wall. It has no idea what customer #4471 has purchased; that data lives in the database. Instead of hallucinating "your recent order," it raises its hand and asks for clarification back to supervisor by calling `query_supervisor`:

> *creative_agent → supervisor:* "What are customer #4471's three most recent purchases?"

The supervisor doesn't know either — so it **delegates the question to the database agent**, gets back `["Trail Runner X2", "Merino socks", "Hydration vest"]`, and relays that straight into the creative agent's paused execution. The creative agent resumes and writes an email that references the customer's *real* gear — all without a single hard-coded step.

That's the whole point: the specialist that needs the data doesn't have to know *who* has it. It just asks, and the supervisor brokers the answer — even pulling in a third agent when needed. (And there's a guardrail: a query is never routed back to the agent that asked it.)

**Under the hood**

```python
async def _solve_query(self, message, agent_name) -> str:
    query_event = self._query_events[agent_name]
    query_event.clear()

    self.agent_tasks[agent_name].status = AgentStatus.WAITING_FOR_QUERY
    self._agent_events[agent_name].set()   # wake the supervisor

    await query_event.wait()               # child pauses, cooperatively
    return self.agent_tasks[agent_name].result
```

The pause is a real `asyncio.Event`, not a busy-wait: the child sets its status to `WAITING_FOR_QUERY`, fires the supervisor's event to wake it, then suspends on `query_event.wait()` — using zero CPU — until the supervisor posts an answer back and unblocks it.

---

## Background execution: stop waiting on slow work

Some jobs are slow. A 30-second data fetch shouldn't freeze your entire orchestration while three other things could be happening.

Mark an agent or tool as `background_runnable=True` and the supervisor gains the ability to **fire it and keep working**:

```python
@supervisor.register_tool("Run a slow data job", background_runnable=True)
async def slow_job(input: str) -> str:
    await asyncio.sleep(30)
    return f"Done: {input}"
```

Behind the scenes this is a proper concurrency model, not a hack:

- Background work is launched with `asyncio.create_task` — true fire-and-forget.
- When it finishes, a `[BACKGROUND UPDATE]` message is **injected into the supervisor's conversation**, so the LLM finds out organically on its next step.
- Need a result *now*? Built-in `wait_for_agent` / `wait_for_tool` tools block until it's ready — and the framework moved from `asyncio.sleep` polling to **event-driven signalling** for this, so there's no wasted CPU.

The cleverest detail is how it prevents **premature completion**. The supervisor finishes a task by calling a special `done` action. But what if there's still a background job running that the final answer depends on? AgentCurie solves this *at the schema level*: while background tasks are pending, it swaps in an alternate output model that **literally removes `done` from the allowed actions**.

```python
response_model = (
    self.AgentOutput
    if (not self.pending_queries and not active_bg_agents and not active_bg_tools)
    else self.AgentOutputWithoutDone   # 'done' is not even an option
)
```

The model *cannot* declare victory early, because the option to do so doesn't exist in its grammar. It also gets a transient reminder ("Background tasks NOT yet complete… do NOT re-launch them") that's injected per-step but never persisted to history — keeping the conversation clean while still steering behavior. That's the difference between *asking* an LLM to behave and *making it impossible to misbehave*.

---

## The conveniences that make it pleasant

**Lazy initialization & persistence.** Child agents aren't constructed until they're first needed — no paying the startup cost for an agent a particular task never touches. Set `persistent=True` and the instance is cached in an active-agents pool and reused across calls within a session; leave it `False` and it's spun up fresh each time.

**A hook system for observability and control.** Inject custom logic anywhere in the loop:

```python
FuncHook(order='before', func=intermediate_logger)            # runs each step, gets the supervisor
AgentHook(order='after', agent_name='creative_agent', func=...) # runs around a specific agent
```

Great for logging, state syncing, guardrails, metrics — without touching the core loop.

**A built-in MCP client.** The Model Context Protocol ecosystem is exploding. AgentCurie ships a client that converts MCP-compatible tool definitions straight into supervisor-level tools, so your agents can reach the whole MCP world with a few lines.

**A token-aware message manager** that counts tokens per message, trims history when it approaches the limit, and even reshapes the conversation for non-function-calling models (and merges consecutive human messages for picky models like DeepSeek-reasoner).

---

## The architecture, at a glance

AgentCurie uses a clean, feature-based modular layout:

```
agentcurie/
├── supervisor/          # The master orchestrator
│   ├── service.py       # SupervisorAgent — the main reasoning loop
│   ├── prompts.py       # System prompts & supervision rules
│   ├── views.py         # FuncHook, AgentHook, AgentContext, status enums
│   └── message_manager/ # Conversation history & token management
├── controller/          # Execution layer
│   ├── service.py       # Dynamic schema generation + dispatch
│   ├── tool/            # Tool registration, registry, execution
│   └── agent/           # Agent registration, registry, execution
└── mcp_client/          # Model Context Protocol integration
```

The **supervisor** thinks. The **controller** acts — it owns the registries, builds the type-guided schemas, and dispatches the chosen tool or agent. The separation is what keeps the orchestration logic framework-agnostic.

---

## Where it's headed

AgentCurie is at **v1.0**, and the roadmap is ambitious in exactly the right ways:

- **Parallel agent execution** with the supervisor acting as a *judge* over competing results.
- **Remote agents and tools** — an MCP-like protocol for agents that live on other machines, turning a local supervisor into a distributed control plane.

---

## The takeaway

The future of agentic systems isn't one model with a thousand tools. It's **many specialists, each excellent at one thing, coordinated by something that reasons about the whole**.

AgentCurie is that something — a framework-agnostic control layer where agents from any ecosystem can collaborate, ask each other questions, run work in the background, and finish only when the job is truly done.

If you've ever felt your multi-agent setup straining under its own weight, give it a try:

```bash
pip install agentcurie
```

⭐ **GitHub:** [github.com/afreediz/agentcurie](https://github.com/afreediz/agentcurie)

I'd love to hear what you build with it — and what you think it should orchestrate next.

*— Afreedi Z*
