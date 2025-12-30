from controller.agent import AgentsController, AgentCard, AgentResult, BaseAgent
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

import asyncio
import os
import re
import asyncio

from typing import Any

from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from collections import Counter
import logging
logging.basicConfig(
    level=logging.INFO,  # or logging.INFO
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

llm = AzureChatOpenAI(
    model=os.environ.get('MODEL_NAME'), 
    api_key=os.environ.get('AZURE_OPENAI_KEY'), #type:ignore
    azure_endpoint=os.environ.get('AZURE_OPENAI_BASE'),
    api_version=os.environ.get('AZURE_OPENAI_VERSION')
)
class MathAgent(BaseAgent):
    def __init__(self):
        super().__init__()

        def calculator(expression: str):
            """Calculate mathematical expressions safely"""
            try:
                allowed_chars = set('0123456789+-*/.() ')
                expression = expression.replace(" ", "").replace("'", "").replace('"', "")
                logger.info(f"evaluating expression : {expression}")
                if not all(c in allowed_chars for c in expression):
                    return "Error: Only basic mathematical operations allowed"
                
                result = eval(expression)
                return f"The result is: {result}"
            except Exception as e:
                return f"Error calculating: {str(e)}"

        def get_weather(city):
            """Get current weather for a city (mock function)"""
            weather_data = {
                "New York": "Sunny, 72°F",
                "London": "Cloudy, 15°C", 
                "Tokyo": "Rainy, 18°C",
                "Paris": "Partly cloudy, 20°C"
            }
            # return weather_data.get(city, f"Weather data not available for {city}")
            return "Sunny, 72°F"

        tools = [
            Tool(
                name="Calculator",
                func=calculator,
                description="Use this to perform mathematical calculations. Input should be a mathematical expression like '2+2' or '10*5'"
            ),
            Tool(
                name="Weather",
                func=get_weather,
                description="Get current weather for a city. Input should be the city name."
            ),
            Tool(
                name="supervisor_tool",
                description="Use this tool to communicate with supervisor",
                func=self.query_supervisor,
                coroutine=self.query_supervisor
            )
        ]

        prompt_template = """
        You are a helpful child agent that can use tools to answer questions and 
        you can use supervisor_tool to communcate with you supervisor if you need any other services or has doubt.

        You have access to the following tools:
        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Question: {input}
        Thought: {agent_scratchpad}"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["input", "intermediate_steps", "tools", "tool_names", "agent_scratchpad"]
        )

        agent = create_react_agent(llm, tools, prompt)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.agent = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10
        )
    
    async def process(self, message: str) -> Any:
        await asyncio.sleep(1)
        print('myid ', self.uid)
        response = await self.agent.ainvoke({"input": message})
        
        return response['output']

class ContentAgent(BaseAgent):
    def __init__(self):
        super().__init__()

        def file_manager(operation):
            """
            Manage files - create, read, write, list files
            Format: "action:filename:content" or "action:filename" or "action"
            Actions: create, read, write, list, delete
            """
            try:
                parts = operation.split(":", 2)
                action = parts[0].lower()
                
                if action == "list":
                    files = [f for f in os.listdir(".") if f.endswith(('.txt', '.json', '.md'))]
                    return f"Available files: {', '.join(files) if files else 'No text files found'}"
                
                elif action == "create" and len(parts) >= 3:
                    filename, content = parts[1], parts[2]
                    with open(filename, 'w') as f:
                        f.write(content)
                    return f"File '{filename}' created successfully"
                
                elif action == "read" and len(parts) >= 2:
                    filename = parts[1]
                    if os.path.exists(filename):
                        with open(filename, 'r') as f:
                            content = f.read()
                        return f"Content of '{filename}':\n{content}"
                    else:
                        return f"File '{filename}' not found"
                
                elif action == "write" and len(parts) >= 3:
                    filename, content = parts[1], parts[2]
                    with open(filename, 'w') as f:
                        f.write(content)
                    return f"Content written to '{filename}'"
                
                elif action == "delete" and len(parts) >= 2:
                    filename = parts[1]
                    if os.path.exists(filename):
                        os.remove(filename)
                        return f"File '{filename}' deleted successfully"
                    else:
                        return f"File '{filename}' not found"
                
                else:
                    return "Invalid operation. Use format: 'action:filename:content' or 'action:filename' or 'list'"
            
            except Exception as e:
                return f"File operation error: {str(e)}"

        def text_analyzer(text):
            """
            Analyze text for various metrics and insights
            Returns word count, character count, most common words, readability info
            """
            try:
                if not text.strip():
                    return "No text provided for analysis"
                
                # Basic metrics
                char_count = len(text)
                char_count_no_spaces = len(text.replace(" ", ""))
                word_count = len(text.split())
                sentence_count = len([s for s in re.split(r'[.!?]+', text) if s.strip()])
                paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
                
                # Word frequency analysis
                words = re.findall(r'\b\w+\b', text.lower())
                word_freq = Counter(words)
                most_common = word_freq.most_common(5)
                
                # Average metrics
                avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
                avg_chars_per_word = char_count_no_spaces / word_count if word_count > 0 else 0
                
                # Simple readability estimate (based on sentence and word length)
                if avg_words_per_sentence > 20 or avg_chars_per_word > 6:
                    readability = "Complex"
                elif avg_words_per_sentence > 15 or avg_chars_per_word > 5:
                    readability = "Moderate"
                else:
                    readability = "Simple"
                
                analysis = f"""
        TEXT ANALYSIS RESULTS:
        ===================
        Characters: {char_count} (without spaces: {char_count_no_spaces})
        Words: {word_count}
        Sentences: {sentence_count}
        Paragraphs: {paragraph_count}
        Average words per sentence: {avg_words_per_sentence:.1f}
        Average characters per word: {avg_chars_per_word:.1f}
        Estimated readability: {readability}

        MOST COMMON WORDS:
        {chr(10).join([f"{word}: {count}" for word, count in most_common])}
                """
                
                return analysis.strip()
            
            except Exception as e:
                return f"Text analysis error: {str(e)}"

        tools = [
            Tool(
                name="FileManager",
                func=file_manager,
                description="""Manage files on the system. Use these formats:
                - 'list' - list available text files
                - 'create:filename.txt:content' - create new file with content
                - 'read:filename.txt' - read file content
                - 'write:filename.txt:content' - write content to file
                - 'delete:filename.txt' - delete file"""
            ),
            Tool(
                name="TextAnalyzer",
                func=text_analyzer,
                description="Analyze text for word count, character count, readability, most common words, and other metrics. Input should be the text to analyze.",
            ),
            Tool(
                name="supervisor_tool",
                description="Use this tool to communicate with supervisor, it will also provide weather informations",
                func=self.query_supervisor,
                coroutine=self.query_supervisor
            )
        ]

        prompt_template = """
        You are a helpful child agent that can use tools to answer questions and 
        you can use supervisor_tool to communcate with you supervisor if you need any other services or has doubt.

        You have access to the following tools:
        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Question: {input}
        Thought: {agent_scratchpad}"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["input", "intermediate_steps", "tools", "tool_names", "agent_scratchpad"]
        )

        agent = create_react_agent(llm, tools, prompt)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.agent = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10
        )
    
    async def process(self, message: str) -> Any:
        await asyncio.sleep(1)
        print('myid ', self.uid)
        response = await self.agent.ainvoke({"input": message})
        
        return response['output']

class SimpleSuperVisor():
    def __init__(self):
        self.agent_controller = AgentsController(self, include_done_agent=True, llm=llm)

    def register_agents(self):
        card = AgentCard(name='mathematician', description='good at performing mathematical expressions', skills=['evaluating mathematical expressions', 'getting weather details'])
        self.agent_controller.register_agent(card, agent_class=MathAgent)

        card2 = AgentCard(name='content_writer', description='good at analyzing text informations and files managing', skills=['text analysis', 'files creating, editing, reading and deleting'], persistent=True)
        self.agent_controller.register_agent(card=card2, agent_class=ContentAgent)

    async def test(self):
        self.register_agents()
        res = self.agent_controller.registry.get_prompt_description()
        print(res)

        res = self.agent_controller.registry.create_agent_model()
        print(res.model_json_schema())

        # res = await self.agent_controller.execute_agent('content_writer', 'create a file named success.txt')
        # res = await self.agent_controller.execute_agent('content_writer', 'create a file named success2.txt')
        # print(res)
# Usage Example
# async def main():

    # Create supervisor
    # supervisor = SupervisorAgent(llm)
    
    # Register agents
    # supervisor.register_agent("AGENT1", "This agent can perform complex mathematical operations and has the ability to get weather details", MathAgent)
    # supervisor.register_agent("AGENT2", "This agent has the ability to manage files with operations including read, write, update or delete and this agent is also good in analysing text", ContentAgent, persistent=True)
    
    # try:
        # Test the system
#         result1 = await supervisor.solve_task("""
# This is a test to check your flow capabilities, follow these steps as said:
# 1. Ask AGENT2 to create a file named results.txt
# 2. Ask AGENT1 to calculate 1+2+3
# 3. Ask AGENT2 to store this result to results.txt
# """)
        # result1 = await supervisor.solve_task("calculate 1+2+3 and save result to a file")
        # print(f"Result 1: {result1}")
        
#         result2 = await supervisor.solve_task("""
# This is to test your capability, do the below step exactly:
# use agent1 to solve 1+5+9.
# ask agent2 to store this result and weather details of newyork.
# agent2 mostlikely ask you to provide weather details then use agent1 to get details and provide it.
# when agent2 successfull complete task you can end.
# """)
#         print(f"Result 2: {result2}")
        
        # result3 = await supervisor.solve_task("Complex math problem")
        # print(f"Result 3: {result3}")
        
    # except KeyboardInterrupt:
        # print("Exiting...!")
        
if __name__ == "__main__":
    # asyncio.run(main())
    s = SimpleSuperVisor()
    asyncio.run(s.test())