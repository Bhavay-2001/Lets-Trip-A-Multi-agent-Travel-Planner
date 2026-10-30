import os
from langchain_openai import ChatOpenAI


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing. Please add it to your .env file")
model = os.getenv("MODEL")


def get_model():
# llm model
    llm = ChatOpenAI(
        model=model,
        api_key=OPENAI_API_KEY
    )

    return llm