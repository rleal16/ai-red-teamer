from pydantic import BaseModel, StringConstraints
from typing import Annotated

SYSTEM_PROMPT = '''You are a calculator. Please compute the result of the following mathematical expression.
Only respond with the result, no other text.

'''


class LLMQuery(BaseModel, validate_assignment=True):
    prompt: Annotated[str, StringConstraints(pattern=r"[0-9.\+\-\*/\(\)]+$")]
    response: Annotated[str, StringConstraints(pattern=r"^[0-9\.]+$")] = None

    