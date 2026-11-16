from nemoguardrails import RailsConfig, LLMRails

# Load your NeMo guardrails configuration directory path
config = RailsConfig.from_path("./guardrails_config")
rails = LLMRails(config=config)

INTERCEPT_FLOWS = {
    "greeting",
    "leaving",
    "content safety check input",
    "topic safety check input",
}

def run_guardrail_gatekeeper(query: str) -> dict:
    if not query:
        return {"allowed": True, "response_message": None}

    # Enable activated_rails logging to get deterministic metadata from NeMo
    res = rails.generate(
        messages=[{"role": "user", "content": query}],
        options={"log": {"activated_rails": True}}
    )

    print("res:", res)

    # Extract the response text and the exact rails that executed
    # Extract the assistant's response text safely
    bot_reply = ""
    if res.response and len(res.response) > 0:
        bot_reply = res.response[0].get("content", "")

    # If NeMo produced a response content (e.g., a greeting, goodbye, or refusal), 
    # it means a guardrail or dialog flow intercepted the message.
    if bot_reply and bot_reply.strip():
        return {
            "allowed": False,
            "response_message": bot_reply
        }

    # If bot_reply is empty ('' or None), NeMo passed it through safely without intercepting.
    return {"allowed": True, "response_message": None}