# src/refusal.py
import random

SAFE_PLAIN_REFUSAL = "I don't know based on the provided sources."

SAFE_SARCASTIC_REFUSALS = [
    "I’d love to answer that, but my sources are giving me the academic equivalent of a blank stare.",
    "Great question — unfortunately the provided sources have chosen to be mysteriously silent.",
    "I looked. The sources looked back. We both agreed: there isn’t enough evidence here.",
    "If guessing were allowed, I’d answer. Sadly, evidence insists on existing.",
    "My sources did a full disappear act on that one — I can’t answer it from what I have.",
]

def refusal_message(sarcastic: bool = False) -> str:
    return random.choice(SAFE_SARCASTIC_REFUSALS) if sarcastic else SAFE_PLAIN_REFUSAL
