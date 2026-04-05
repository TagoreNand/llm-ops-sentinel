"""Token cost calculator for supported models."""
from app.config import get_settings

settings = get_settings()


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return total cost in USD for a given LLM call."""
    costs = settings.model_costs.get(model, {"input": 0.001, "output": 0.002})
    return round(
        (input_tokens / 1000) * costs["input"]
        + (output_tokens / 1000) * costs["output"],
        6,
    )
