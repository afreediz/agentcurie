from controller.agent.registery.views import AgentModel
from pydantic import RootModel

def unwrap_agent_model(choice: AgentModel | RootModel) -> AgentModel:
    if isinstance(choice, RootModel):
        return choice.root
    return choice