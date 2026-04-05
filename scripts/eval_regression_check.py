"""
CI eval regression gate.

Runs the evaluator against golden prompt/response pairs.
Exits with code 1 if average score is below the threshold.
"""
import asyncio
import sys
from app.evaluators.judge import evaluate

GOLDEN_PAIRS = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is 2 + 2?", "2 + 2 equals 4."),
    ("What is HTTP?", "HTTP stands for Hypertext Transfer Protocol, used for web communication."),
]

THRESHOLD = 0.65


async def run():
    scores = []
    for prompt, response in GOLDEN_PAIRS:
        result = await evaluate(prompt, response)
        scores.append(result.overall_score)
        print(f"  [{result.overall_score:.3f}] {prompt[:50]}")

    avg = sum(scores) / len(scores)
    print(f"\nAverage eval score: {avg:.3f} (threshold: {THRESHOLD})")

    if avg < THRESHOLD:
        print("FAIL: Eval regression detected. Check recent prompt or model changes.")
        sys.exit(1)
    else:
        print("PASS: Eval scores meet threshold.")


if __name__ == "__main__":
    asyncio.run(run())
