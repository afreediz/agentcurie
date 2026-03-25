import json
import asyncio
from agentcurie import SupervisorAgent, AgentCard, ToolResult
from examples.env import config
from .agent1 import CreativeAgent
from .agent2 import DBAgent
from .agent2 import fake_db
import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    api_key=config.openai_key,#type:ignore
    model=config.model
)

# AGENT CARDS
card_first = AgentCard(
    name='database_agent', 
    description='Can perform database operations including storing and querying datas', 
    skills=['store data','get data','list data']
)
card_second = AgentCard(
    name='creative_agent', 
    description='Can write poems and perform calculations', 
    skills=['write poem','does calculations like add, subtract, multiply, divide'], 
    persistent=True
)

# Usage Example
async def main():
    # Create supervisor
    supervisor = SupervisorAgent(llm=llm)
    
    # Register agents
    supervisor.register_agent(agent_card=card_first, agent_class=DBAgent)
    supervisor.register_agent(agent_card=card_second, agent_class=CreativeAgent)

    @supervisor.register_tool('Use to get weather details of any place')
    def get_weather(city: str):
        """Get current weather for a city (mock function)"""
        weather_data = {
            "New York": "Sunny, 72°F",
            "London": "Cloudy, 15°C",
            "Tokyo": "Rainy, 18°C",
            "Paris": "Partly cloudy, 20°C"
        }
        # return weather_data.get(city, f"Weather data not available for {city}")

        return "Sunny, 72°F"
    
    @supervisor.register_tool("Tester tool")
    def tester_tool(a:int):
        print("Tester tool called ====")
        return ToolResult(content="exectured successfully")
    
    @supervisor.register_tool("Tester tool")
    def tester_tool2(a:int, b:int):
        print("Tester tool 2 called ====")
        return ToolResult(content="exectured successfully")

    try:
        # Test the system
#         result1 = await supervisor.solve("""
# Call both tester tools
# """)
        
        result1 = await supervisor.solve("""
This task is to evaluate your capability to follow instructions, do as exact:
1. Ask creative agent to calculate x + 10 and tell him you can find x by querying to supervisor (do not tell your the supervisor, just ask it to use tool).
2. After first step creative agent will ask you value for x, you should ask db agent to provide a random value.
3. assign this random value to creative agent query
4. After getting the result from creative agent, tell db agent to insert this value to db
5, Call tester tool 1 with input value as x and tester tool 2 with input vale as x*2, x*3
5. close by ensuring all success
""")
        
        result1.resources["resources"] = fake_db

        with open(r'./temp.result.json', 'w') as f:
            json.dump(result1.model_dump(), f, indent=2)

    except KeyboardInterrupt:
        print("Exiting...!")
        
if __name__ == "__main__":
    asyncio.run(main())