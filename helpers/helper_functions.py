import spacy
from typing import Any
import json

from langchain_core.messages import ( 
    HumanMessage,
    SystemMessage
)

from llm_model.llm_model import get_model

nlp = spacy.load("en_core_web_sm")
llm = get_model()


def extract_destination(query: str) -> str | None:
    doc = nlp(query)
    
    # Filter entities for GPE (Geopolitical Entity: cities, states, countries) or LOC (Locations)
    locations = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC")]
    
    return locations[0] if locations else None


def _llm_text(system_prompt: str, user_prompt: str) -> str:
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return str(response.content)

def _json_from_llm(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object returned by the model."""
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON object.")

    return json.loads(text[start : end + 1])


# Known Agents
KNOWN_AGENTS = {
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
}

# Agent Order
AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]