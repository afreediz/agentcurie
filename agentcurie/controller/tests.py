#type:ignore
import os
from pydantic import BaseModel
from controller import Controller, ToolResult, AgentCard, BaseAgent
from controller.views import AgentOutput
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import AzureChatOpenAI
llm = AzureChatOpenAI(
    model=os.environ.get('MODEL_NAME'), 
    api_key=os.environ.get('AZURE_OPENAI_KEY'), #type:ignore
    azure_endpoint=os.environ.get('AZURE_OPENAI_BASE'),
    api_version=os.environ.get('AZURE_OPENAI_VERSION')
)

class SimpleToolModel(BaseModel):
    name: str
    age: int

class SimpleToolModel2(BaseModel):
    bank: str
    pan: int

class SimpleAgent():
    def __init__(self) -> None:
        self.controller = Controller(supervisor=self, llm=llm)

    def test(self):
        @self.controller.tool("simple description")
        async def simple_tool(param: SimpleToolModel):
            return ToolResult(content='simple result')
        
        @self.controller.tool('simple description 2')
        async def simple_tool_2(param: SimpleToolModel2):
            return ToolResult(content="simple result 2")
        
        card = AgentCard(name='simple_agent', description='simple agent to manage files', skills=['file manager'])
        class SimpleChildAgent(BaseAgent):
            def __init__(self) -> None:
                print('Initialized child agent')

        card2 = AgentCard(name='simple_agent_2', description='simple agent to do math', skills=['mathematical operations'])
        class SimpleChildAgent2(BaseAgent):
            def __init__(self) -> None:
                print('Initialized child agent 2')

        self.controller.register_agent(agent_card=card, agent_class=SimpleChildAgent)
        self.controller.register_agent(agent_card=card2, agent_class=SimpleChildAgent2)

        sys_prompt = self.controller.get_prompt_description()

        choice_model = self.controller.create_choice_model()
        # print(choice_model.model_json_schema())

        agent_out_model = AgentOutput.type_with_custom_tools_and_agents(choice_model) #type:ignore

        # message = "Call the second tool with bank as sbi and pan as 1234"
        message = "Call a agent that can go math operations and ask it to do 1+1"
        res = llm.with_structured_output(agent_out_model, method="function_calling").invoke(input=[
            SystemMessage(content=sys_prompt),
            HumanMessage(message)
        ]) #type:ignore

        print(res, '\n\n', res.model_dump())

        print(res, '\n 2 : ', res.action.choice.get_type(), '\n 3 :', type(res.action))

        print(res.get_choice())
        # print(type(res), res.choice.get_type(), res.model_dump_json()) #type:ignore

if __name__ == "__main__":
    agent = SimpleAgent()
    agent.test()