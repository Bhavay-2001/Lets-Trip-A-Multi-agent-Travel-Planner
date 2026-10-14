from langchain.messages import AIMessage
from langchain_typesafe import Noul, TypeSafeClassifier

from agent.schema import TravelState
from helpers.helper_functions import AGENT_ORDER, KNOWN_AGENTS, _json_from_llm, _llm_text

from prompts.prompts import agent_questions, guardrail_prompt


JEV_AGENT_THRESHOLD = 0.70

jev_classifier = TypeSafeClassifier()


# =========================
# Supervisor Agent + Input Guardrail
# =========================
def supervisor_agent(state: TravelState):
    query = state["user_query"]
    llm_calls = state.get("llm_calls", 0)

    # Fail open on parser/model errors so a temporary JSON-format issue does not
    # break the original travel-planning behavior.
    try:
        guardrail_raw = _llm_text(
            "You are the input guardrail for a travel-planning application. "
            "Return strict JSON only.",
            guardrail_prompt,
        )
        guardrail_result = _json_from_llm(guardrail_raw)
        allowed = bool(guardrail_result.get("allowed", True))
        guardrail_reason = str(guardrail_result.get("reason", "")).strip()
        llm_calls += 1
    except Exception as exc:
        print(f"Guardrail fallback used: {exc}")
        allowed = True
        guardrail_reason = "Guardrail validation fallback allowed the request."

    if not allowed:
        reason = guardrail_reason or (
            "TripMate AI can only help with travel-planning requests. "
            "Please ask about a destination, flight, hotel, weather, budget, "
            "or itinerary."
        )
        return {
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "selected_agents": [],
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [AIMessage(content=f"Guardrail blocked request: {reason}")],
            "llm_calls": llm_calls,
        }

    # JEV Based classification
    try:
        jev_result = jev_classifier.invoke({
            "state": {
                "user_query": query,
                "available_agents": list(KNOWN_AGENTS),
            },
            "questions": agent_questions,
        })

        # Each Noul returns P(yes) for its corresponding agent.
        agent_probabilities = {
            agent: float(jev_result.nouls[agent].noul)
            for agent in agent_questions
        }

        selected_agents = [
            agent
            for agent in AGENT_ORDER
            if agent in agent_probabilities
            and agent_probabilities[agent] >= JEV_AGENT_THRESHOLD
        ]

        # Always retain the itinerary agent.
        selected_agents.append("itinerary_agent")

        # Remove duplicates while preserving order.
        selected_agents = list(dict.fromkeys(selected_agents))

        # JEV does not generate free-form reasoning like an LLM.
        # Store the probability scores for inspection instead.
        reasoning = (
            "JEV agent-selection probabilities: "
            + ", ".join(
                f"{agent}={probability:.3f}"
                for agent, probability in agent_probabilities.items()
            )
            + f". Selection threshold={JEV_AGENT_THRESHOLD:.2f}."
        )

        print("JEV Reasoning:", reasoning)

    except Exception as exc:
        print(f"JEV supervisor fallback used: {exc}")

        # Preserve the original safe fallback.
        selected_agents = AGENT_ORDER.copy()
        reasoning = (
            "JEV selection failed; all agents were selected as a fallback."
        )

    return {
        "guardrail_allowed": True,
        "guardrail_reason": guardrail_reason,
        "selected_agents": selected_agents,
        "supervisor_reasoning": reasoning,
        "messages": [
            AIMessage(content="JEV created the agent plan.")
        ],
        "llm_calls": llm_calls,
    }
