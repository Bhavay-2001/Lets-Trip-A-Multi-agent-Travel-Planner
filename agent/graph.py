from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg
import os
from dotenv import load_dotenv

from agent.agents import budget_agent, final_agent, flight_agent, guardrail_blocked_agent, hotel_agent, human_approval_agent, itinerary_agent, weather_agent
from agent.helper import ROUTE_MAP, route_after_agent, route_from_supervisor
from agent.schema import TravelState
from agent.supervisor_agent import supervisor_agent

load_dotenv()

POSTGRES_DATABASE_URL = os.getenv("POSTGRES_DATABASE_URL")
conn = psycopg.connect(POSTGRES_DATABASE_URL, autocommit=True)
checkpointer = PostgresSaver(conn)
checkpointer.setup()

# =========================
# Build Graph
# =========================
builder = StateGraph(TravelState)

builder.add_node("supervisor", supervisor_agent)
builder.add_node("guardrail_blocked", guardrail_blocked_agent)
builder.add_node("flight_agent", flight_agent)
builder.add_node("hotel_agent", hotel_agent)
builder.add_node("weather_agent", weather_agent)
builder.add_node("budget_agent", budget_agent)
builder.add_node("itinerary_agent", itinerary_agent)
builder.add_node("human_approval", human_approval_agent)
builder.add_node("final_agent", final_agent)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route_from_supervisor, ROUTE_MAP)

builder.add_conditional_edges(
    "flight_agent", route_after_agent("flight_agent"), ROUTE_MAP
)
builder.add_conditional_edges(
    "hotel_agent", route_after_agent("hotel_agent"), ROUTE_MAP
)
builder.add_conditional_edges(
    "weather_agent", route_after_agent("weather_agent"), ROUTE_MAP
)
builder.add_conditional_edges(
    "budget_agent", route_after_agent("budget_agent"), ROUTE_MAP
)

builder.add_edge("itinerary_agent", "human_approval")
builder.add_edge("human_approval", "final_agent")
builder.add_edge("final_agent", END)
builder.add_edge("guardrail_blocked", END)

graph = builder.compile()