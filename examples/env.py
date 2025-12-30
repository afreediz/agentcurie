import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()

@dataclass
class CONFIG:
    openai_key: str
    model: str


openai_key = os.environ.get("OPENAI_KEY")
model = os.environ.get("MODEL")

if openai_key is None:
    raise EnvironmentError("OPENAI_KEY is not set")

if model is None:
    raise EnvironmentError("MODEL is not set")

config = CONFIG(
    openai_key=openai_key,
    model=model
)
