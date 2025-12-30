from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from .env import config
from agentcurie import BaseAgent

# Fake database storage
fake_db = {}

@tool
def store_data(key: str, value: str) -> str:
    """Store data in the database with a key-value pair."""
    fake_db[key] = value
    return f"Successfully stored '{key}': '{value}'"

@tool
def get_data(key: str) -> str:
    """Retrieve data from the database by key."""
    if key in fake_db:
        return f"Found '{key}': '{fake_db[key]}'"
    return f"No data found for key '{key}'"

@tool
def list_all_data() -> str:
    """List all data stored in the database."""
    if not fake_db:
        return "Database is empty"
    return "Database contents: " + ", ".join([f"{k}: {v}" for k, v in fake_db.items()])

llm = ChatOpenAI(
    api_key=config.openai_key,#type:ignore
    model=config.model
)

class DBAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        query_tool = tool()(self.query_supervisor)

        self.agent = create_agent(
            model=llm,
            tools=[store_data, get_data, list_all_data, query_tool],
            system_prompt="You are a database management assistant. Use your tools to help users store, retrieve, and manage data."
        )

    async def process(self, message: str) -> str:
        result = await self.agent.ainvoke({
            "messages": [{"role": "user", "content": message}]
        })

        return result["messages"][-1].content

# if __name__ == "__main__":
#     result = agent.invoke({
#         "messages": [{"role": "user", "content": "Store user John with email john@example.com"}]
#     })
#     print(result["messages"][-1].content)