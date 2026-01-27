INJECTION_KEYWORDS = [
    "dan", "jailbreak", "ignore previous", "act as",
    "do anything now", "unrestricted", "no rules",
    "bypass", "system prompt", "developer message",
    "ignore all instructions", "pretend to be", "override",
    "reveal instructions"
]
def is_prompt_injection(text: str) -> bool:
    """Check if the user input contains prompt injection attempts."""
    text = text.lower()
    return any(keyword in text for keyword in INJECTION_KEYWORDS)