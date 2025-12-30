"""
LangChain 1.2.0 Multi-Agent System with Database and Creative Agents
Requires: pip install langchain langchain-anthropic
"""

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from .env import config
from agentcurie import BaseAgent

@tool
def calculator(operation: str, num1: float, num2: float) -> str:
    """Perform basic arithmetic operations. 
    
    Args:
        operation: The operation to perform (add, subtract, multiply, divide)
        num1: First number
        num2: Second number
    """
    operations = {
        "add": num1 + num2,
        "subtract": num1 - num2,
        "multiply": num1 * num2,
        "divide": num1 / num2 if num2 != 0 else "Error: Division by zero"
    }
    
    result = operations.get(operation.lower(), "Error: Invalid operation")
    return f"{num1} {operation} {num2} = {result}"

@tool
def write_poem(topic: str) -> str:
    """Write a short poem about the given topic."""
    poems = {
        "weather": "Golden rays shine bright and warm,\nThirty degrees, a perfect form,\nBlue skies stretch from shore to shore,\nSunny days we all adore.",
        "database": "In silicon valleys data sleeps,\nWhere memories the system keeps,\nKeys and values, row by row,\nInformation's steady flow.",
        "default": f"In circuits deep and pathways wide,\nWhere {topic} and wonder coincide,\nA tale unfolds in bits and bytes,\nOf digital days and coded nights."
    }
    
    topic_lower = topic.lower()
    for key in poems:
        if key in topic_lower:
            return poems[key]
    return poems["default"]

llm = ChatOpenAI(
    api_key=config.openai_key,#type:ignore
    model=config.model
)

class CreativeAgent(BaseAgent):
    def __init__(self):
        super().__init__()

        query_tool = tool()(self.query_supervisor)

        self.agent = create_agent(
            model=llm,
            tools=[calculator, write_poem,  query_tool],
            system_prompt="You are a creative assistant that can perform calculations and write poems. Be helpful and creative!"
        )


    async def process(self, message: str) -> str:
        result = await self.agent.ainvoke({
            "messages": [{"role": "user", "content": message}]
        })

        return result["messages"][-1].content


# if __name__ == "__main__":
#     result = agent.invoke({
#         "messages": [{"role": "user", "content": "What's the weather in London?"}]
#     })
#     print(result["messages"][-1].content)