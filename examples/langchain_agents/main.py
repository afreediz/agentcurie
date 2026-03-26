import json
import asyncio
from agentcurie import SupervisorAgent, AgentCard, ToolResult
from examples.env import config
from .agent1 import CreativeAgent
from .agent2 import DBAgent
from .agent2 import fake_db
from .agent3 import SheffAgent
import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
for _noisy in ('httpx', 'httpcore', 'openai', 'langsmith', 'urllib3'):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

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
card_third = AgentCard(
    name='sheff_agent',
    description='Can cook food, make juice, tea of snacks',
    skills=['find_recipe', 'cook'],
    persistent=False,
    background_runnable=True
)

# Usage Example
async def main():
    # Create supervisor
    supervisor = SupervisorAgent(llm=llm)
    
    # Register agents
    supervisor.register_agent(agent_card=card_first, agent_class=DBAgent)
    supervisor.register_agent(agent_card=card_second, agent_class=CreativeAgent)
    supervisor.register_agent(agent_card=card_third, agent_class=SheffAgent)

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
    
    @supervisor.register_tool("Tester tool 1", background_runnable=True)
    async def tester_tool_1(a:int):
        print("\n\nTester tool called ====\n")
        await asyncio.sleep(20)
        print("\n\n tester tool completed")
        return ToolResult(content="exectured successfully")
    
    @supervisor.register_tool("Tester tool 2")
    def tester_tool2(a:int, b:int):
        print("\n\nTester tool 2 called ====\n")
        return ToolResult(content="exectured successfully")

    try:
        # Test the system
        result1 = await supervisor.solve("""
Call both tester tool 1 in background then call tester tool 2 and done
""")
        
#         result1 = await supervisor.solve("""
# This task is to evaluate your capability to follow instructions, do as exact:
# 1. Tell sheff to make a tea (in background)
# 2. Ask creative agent to calculate x + 10 and tell him you can find x by querying to supervisor (do not tell your the supervisor, just ask it to use tool).
# 3. After first step creative agent will ask you value for x, you should ask db agent to provide a random value.
# 4. assign this random value to creative agent query
# 5. Get the tea from sheff here only and do the below tasks
# 6. After getting the result from creative agent, tell db agent to insert this value to db
# 7, Call tester tool 1 with input value as x and tester tool 2 with input vale as x*2, x*3
# 8. close by ensuring all success
# """)
        
        result1.resources["resources"] = fake_db

        with open(r'./temp.result.json', 'w') as f:
            json.dump(result1.model_dump(), f, indent=2)

    except KeyboardInterrupt:
        print("Exiting...!")
        
if __name__ == "__main__":
    asyncio.run(main())