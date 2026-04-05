"""
Seed the database with a golden evaluation set.

Run this once after first boot:
  docker-compose run --rm app python scripts/seed_golden_set.py
"""
import asyncio
from app.database import AsyncSessionLocal, init_db, LLMCall
from app.core.hasher import hash_prompt

GOLDEN_PAIRS = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is 2 + 2?", "2 + 2 equals 4."),
    ("What programming language is known for its simplicity and readability?", "Python is widely known for its simplicity and readable syntax."),
    ("Explain what an API is.", "An API (Application Programming Interface) is a set of rules that allows different software applications to communicate with each other."),
    ("What does HTTP stand for?", "HTTP stands for Hypertext Transfer Protocol."),
]


async def seed():
    await init_db()
    async with AsyncSessionLocal() as db:
        for prompt, response in GOLDEN_PAIRS:
            call = LLMCall(
                prompt_hash=hash_prompt(prompt),
                model="gpt-4o",
                prompt_text=prompt,
                response_text=response,
                input_tokens=len(prompt.split()),
                output_tokens=len(response.split()),
                cost_usd=0.0,
                latency_ms=0.0,
                metadata_={"golden": True},
            )
            db.add(call)
        await db.commit()
    print(f"Seeded {len(GOLDEN_PAIRS)} golden examples.")


if __name__ == "__main__":
    asyncio.run(seed())
