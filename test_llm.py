from llm.azure_openai import get_openai_client
from pydantic import BaseModel
import os


class TestResponse(BaseModel):
    answer: str
    number: int


client = get_openai_client()

model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

print("Deployment:", model)

response = client.chat.completions.parse(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "Return answer='Azure works' and number=42."
        }
    ],
    response_format=TestResponse
)

print(response.choices[0].message.parsed)