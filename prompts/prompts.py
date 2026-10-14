

from langchain_typesafe import Noul

from agent.schema import TravelState


def guardrail_prompt(query: str):
    return f"""
    Determine whether the following request belongs to travel planning or travel
    information. Valid requests can include destinations, flights, hotels, weather,
    budgets, visas, transportation, sightseeing, food, packing, or itineraries.

    Block clearly unrelated requests and requests asking for harmful or illegal
    instructions. Do not block a valid travel request merely because some details
    are missing.

    Return strict JSON only:
    {{
    "allowed": true,
    "reason": ""
    }}

    User request:
    {query}
    """


def agent_questions():
    return {
            "flight_agent": Noul(
                instructions=(
                    """Does the user's travel request require flight information, 
                    airports, airlines, routes, or airfare?"""
                )
            ),
            "hotel_agent": Noul(
                instructions=(
                    """Does the user's travel request require hotel or 
                    accommodation information?"""
                )
            ),
            "weather_agent": Noul(
                instructions=(
                    """Does the user's travel request require weather, climate, 
                    seasonal conditions, or weather-related packing advice?"""
                )
            ),
            "budget_agent": Noul(
                instructions=(
                    """Does the user's travel request require budget analysis, 
                    cost estimation, affordability, or price comparisons?"""
                )
            ),
        }


def flight_agent_prompt(query: str, airport_data: str, airline_data: str):
    return f"""
            You are a travel flight expert.

            User query: {query}

            Airport Information: {airport_data}

            Airline Information: {airline_data}

            Generate:
            1. Likely departure airport
            2. Likely arrival airport
            3. Airlines serving this route
            4. typical flight duration
            5. Estimated airfare range
            6. Peak season pricing warning
            7. Booking advice

            Return concise travel guidance.
            """


def budget_agent_prompt(state: TravelState):
    return f"""
            Analyze whether this trip is realistic for the user's budget.

            User Query:
            {state['user_query']}

            Flight Results:
            {state.get('flight_results', '')}

            Hotel Results:
            {state.get('hotel_results', '')}

            Weather Results:
            {state.get('weather_results', '')}

            Return:
            1. Estimated cost categories
            2. Budget risk areas
            3. Money-saving suggestions
            4. Overall feasibility

            If exact live prices are unavailable, clearly label estimates as approximate.
            """


def itinerary_agent_prompt(state: TravelState):
    return f"""
            Create a complete travel itinerary.

            User Query:
            {state['user_query']}

            Flight Results:
            {state.get('flight_results', '')}

            Hotel Results:
            {state.get('hotel_results', '')}

            Weather Results:
            {state.get('weather_results', '')}

            Budget Results:
            {state.get('budget_results', '')}

            Make the itinerary practical, budget-aware and easy to follow.
            """


def final_agent_prompt(state: TravelState, review_instruction: str):
    return f"""
            Generate the final travel response for the user.

            Human Review:
            {review_instruction}

            User Request:
            {state['user_query']}

            Flights:
            {state.get('flight_results', '')}

            Hotels:
            {state.get('hotel_results', '')}

            Weather:
            {state.get('weather_results', '')}

            Budget Analysis:
            {state.get('budget_results', '')}

            Draft Itinerary:
            {state.get('itinerary', '')}

            Format the final answer beautifully using these sections:
            1. Trip Summary
            2. Flight Information
            3. Hotel Suggestions
            4. Weather Information
            5. Day-by-Day Itinerary
            6. Estimated Budget
            7. Final Recommendations

            Important:
            - Be clear and practical.
            - Mention that live flight APIs may not provide ticket prices when pricing is unavailable.
            - Include weather-based travel advice.
            - Keep the response useful for real travel planning.
            - Incorporate the human feedback when revision was requested.
            """