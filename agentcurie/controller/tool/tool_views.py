from pydantic import BaseModel

class Done(BaseModel):
    is_done: bool
    summary: str