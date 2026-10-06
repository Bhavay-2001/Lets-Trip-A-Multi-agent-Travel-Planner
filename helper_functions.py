from langchain_groq import ChatGroq
import os


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file")
model = os.getenv("MODEL")

# llm model
llm = ChatGroq(
    model=model,
    api_key=GROQ_API_KEY
)

def extract_destination(user_query: str):

    prompt = f"""
    Extract only the destination city or country from user query.

    Query: {user_query}

    Return the destination name.
    """

    response = llm.invoke(prompt)

    return response.content.strip()