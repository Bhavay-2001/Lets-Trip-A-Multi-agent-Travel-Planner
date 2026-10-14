


# =========================
# Guardrail blocked response
# =========================
import asyncio

from langchain.messages import AIMessage
from langgraph.types import Command, interrupt

from agent.schema import TravelState
from helpers.helper_functions import _llm_text, extract_destination
from my_mcp.mcp_tools import aviationstack_mcp, tavily_mcp_search, weather_mcp_search
from prompts.prompts import budget_agent_prompt, final_agent_prompt, flight_agent_prompt, itinerary_agent_prompt


def guardrail_blocked_agent(state: TravelState):

    reason = state.get("final_response") or state.get("guardrail_reason") or (
        "This request was blocked by the travel input guardrail."
    )
    return {
        "final_response": reason,
        "messages": [AIMessage(content=reason)],
    }


# =========================
# Flight agent
# =========================

def flight_agent(state: TravelState):

    query = state['user_query']

    try:
        airports = asyncio.run(
            aviationstack_mcp(
                "list_airports"
            )
        )

        airlines = asyncio.run(
            aviationstack_mcp(
                "list_airlines"
            )
        )

        prompt = flight_agent_prompt(
            query=query,
            airport_data=airports[:3000],
            airline_data=airlines[:3000]
        )

        response = _llm_text(
            system_prompt="You are an expert travel flight planner.",
            user_prompt=prompt
        )

        flight_data = response.content

    except Exception as e:
        flight_data = f"Flight information unavailable for {state['user_query']}: {str(e)}"

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(
                content="Flight recoomendations generated"
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# Hotel agent
# =========================
def hotel_agent(state: TravelState):

    query = f"Best hotels for {state['user_query']}"

    hotel_results = asyncio.run(tavily_mcp_search(query))

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content="Hotel information fetched.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# Weather agent
# =========================
def weather_agent(state: TravelState):

    city_name = extract_destination(state["user_query"])

    weather_data = asyncio.run(weather_mcp_search(city_name, "weather"))

    forecast_data = asyncio.run(weather_mcp_search(city_name, "forecast"))

    return {
        "weather_results": f"""
        Current weather: {weather_data}

        Forecast: {forecast_data}
        """,
        "messages": [
            AIMessage(content="Weather information fetched.")
        ]
    }


# =========================
# Budget Agent 
# =========================
def budget_agent(state: TravelState):

    prompt = budget_agent_prompt(state)

    response = _llm_text(
        system_prompt="You are a practical travel budget analyst.",
        user_prompt=prompt
    )

    return {
        "budget_results": response.content,
        "messages": [AIMessage(content="Budget assessment generated.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Itinerary Agent
# =========================
def itinerary_agent(state: TravelState):

    prompt = itinerary_agent_prompt(state)
    response = _llm_text(
        system_prompt="You are an expert travel planner and guide.",
        user_prompt=prompt
    )
    
    approval_request = (
        "Please review the generated draft itinerary. Approve it to create the "
        "final polished plan, or provide feedback for revision."
    )

    return {
        "itinerary": response.content,
        "approval_request": approval_request,
        "messages": [AIMessage(content="Draft itinerary created for human review.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Human-in-the-Loop approval
# =========================
def human_approval_agent(state: TravelState):

    review = interrupt(
        {
            "question": "Do you approve this itinerary?",
            "draft_itinerary": state.get("itinerary", ""),
            "approval_request": state.get("approval_request", ""),
            "selected_agents": state.get("selected_agents", []),
            "supervisor_reasoning": state.get("supervisor_reasoning", ""),
            "expected_response": {
                "approved": True,
                "feedback": "Optional revision feedback",
            },
        }
    )

    approved = bool(review.get("approved", False))
    human_feedback = str(review.get("feedback", "")).strip()

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [AIMessage(content="Human approval step completed.")],
    }


# =========================
# Final Response Agent - original format kept, HITL feedback added
# =========================
def final_agent(state: TravelState):
    
    if state.get("approved", False):
        review_instruction = (
            "The user approved the draft. Preserve its decisions while polishing it."
        )
    else:
        review_instruction = f"""
        The user requested a revision. Apply this feedback carefully:
        {state.get('human_feedback', '') or 'Improve the draft before finalizing it.'}
        """

    final_prompt = final_agent_prompt(
        state,
        review_instruction
    )

    response = _llm_text(
        system_prompt="You are a professional AI travel booking assistant.",
        user_prompt=final_prompt
    )

    return {
        "final_response": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }
