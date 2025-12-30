from controller import ToolsController
from pydantic import BaseModel
import asyncio

class SimpleModel(BaseModel):
    query: str

async def main():
    controller = ToolsController()

    @controller.registry.tool("Test func1", param_model=SimpleModel)
    async def func1(params: SimpleModel):
        print('execute func1')

    @controller.registry.tool("Test func2", param_model=SimpleModel)
    async def func2(params: SimpleModel):
        print("executed func2", params)

    # get description of all tools with types specified
    res = controller.registry.get_prompt_description()
    print('desc ',res)

    # create union of tools models
    res = controller.registry.create_tool_model()
    print('param model ', res.model_json_schema())

    # execute tools
    res = await controller.registry.execute_tool('func1', {'query':'hi'})
    print(res)

if __name__ == "__main__":
    asyncio.run(main())
else:
    raise ValueError("tests cannot be imported")