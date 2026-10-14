import os
from langchain_groq import ChatGroq



GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file")
model = os.getenv("MODEL")


def get_model():
# llm model
    llm = ChatGroq(
        model=model,
        api_key=GROQ_API_KEY
    )

    return llm