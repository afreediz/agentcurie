"""
LangChain 1.2.0 Multi-Agent System with Database and Creative Agents
Requires: pip install langchain langchain-anthropic
"""

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from examples.env import config
from agentcurie import BaseAgent
import asyncio
import logging
logger = logging.getLogger(__name__)

@tool
def find_recipe(operation: str, num1: float, num2: float) -> str:
    "finds food recipe from internet"
    return "Guess by yourself"

@tool
def cook_recipe(topic: str) -> str:
    "makes the requested food"
    return "Cooked and delivered"

llm = ChatOpenAI(
    api_key=config.openai_key,#type:ignore
    model=config.model
)

class SheffAgent(BaseAgent):
    def __init__(self):
        super().__init__()

        query_tool = tool()(self.query_supervisor)

        self.agent = create_agent(
            model=llm,
            tools=[find_recipe, cook_recipe,  query_tool],
            system_prompt="You are a sheff who knows all recipe and cooks delicious food."
        )


    async def process(self, message: str) -> str:
        print("\n\nCook called ----\n\n")
        await asyncio.sleep(10)
        print("\n\nCook completed ----\n\n")
        return "Cooked and served to the table"


# if __name__ == "__main__":
#     result = agent.invoke({
#         "messages": [{"role": "user", "content": "What's the weather in London?"}]
#     })
#     print(result["messages"][-1].content)