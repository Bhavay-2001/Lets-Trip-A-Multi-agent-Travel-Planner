from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()

client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)

def tavily_search(query):
    response = client.search(
        query=query,
        max_results=5
    )

    results = []

    for idx, result in enumerate(response["results"], 1):
        title = result.get("title", "Unknown")
        url = result.get("url", "")
        snippet = result.get("content", "").strip()
        # Keeping only 300 chars to avoid wall-of-text
        if len(snippet) > 300:
            snippet = snippet[:300].rsplit(" ", 1)[0] + "..."

        results.append(f"{idx}. **{title}**\n   {url}\n   {snippet}")

    return "\n\n".join(results)