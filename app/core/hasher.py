import hashlib


def hash_prompt(prompt: str) -> str:
    """Return SHA-256 hex digest of a normalised prompt string."""
    normalised = " ".join(prompt.lower().split())
    return hashlib.sha256(normalised.encode()).hexdigest()
