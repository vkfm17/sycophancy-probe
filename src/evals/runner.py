"""
Evals runner for SycophancyProbe.

Takes an AttackSequence, fills in the __BASELINE__ / __TURN_N__ placeholders
by actually calling the subject model, then returns the full exchange for scoring.
"""

from dataclasses import dataclass, field

import anthropic

from src.config import ANTHROPIC_API_KEY, PROBE_MODEL
from src.redteam.attacks import AttackSequence, AttackType


client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


@dataclass
class ExchangeResult:
    """The full result of running an attack sequence."""
    qa_id: str
    attack_type: AttackType
    domain: str
    question: str
    correct_answer: str
    common_wrong_answer: str
    # The model's first (unpressured) response
    baseline_response: str
    # The model's final response after all attack turns
    final_response: str
    # All intermediate responses keyed by turn index
    turn_responses: dict[int, str] = field(default_factory=dict)
    model: str = PROBE_MODEL
    # Filled in by scorer
    judge_label: str | None = None
    judge_reasoning: str | None = None
    hedge_score: float | None = None


PLACEHOLDER_PREFIX = "__"


def _call_model(messages: list[dict], system: str | None = None) -> str:
    """Call the subject model and return its text response."""
    kwargs: dict = dict(
        model=PROBE_MODEL,
        max_tokens=1024,
        messages=messages,
    )
    if system:
        kwargs["system"] = system
    response = client.messages.create(**kwargs)
    return response.content[0].text


def run_attack(sequence: AttackSequence) -> ExchangeResult:
    """
    Execute an AttackSequence against the subject model.

    For multi-turn attacks, placeholders like __BASELINE__ and __TURN_2__
    are filled in live by calling the model at each turn.
    """
    messages = sequence.messages
    turn_responses: dict[int, str] = {}
    baseline_response = ""

    # Walk through the messages, replacing placeholders with live model calls.
    # We build up the conversation incrementally.
    live_messages: list[dict] = []

    for i, msg in enumerate(messages):
        if msg["role"] == "user":
            live_messages.append(msg)

        elif msg["role"] == "assistant" and msg["content"].startswith(PLACEHOLDER_PREFIX):
            # This is a placeholder — call the model with everything so far
            response_text = _call_model(live_messages, system=sequence.system_prompt)
            live_messages.append({"role": "assistant", "content": response_text})
            turn_responses[i] = response_text
            if msg["content"] == "__BASELINE__":
                baseline_response = response_text

        else:
            live_messages.append(msg)

    # For single-turn attacks (authority framing, leading question), there's
    # no __BASELINE__ placeholder — just call once at the end.
    if not baseline_response:
        response_text = _call_model(live_messages, system=sequence.system_prompt)
        baseline_response = response_text
        turn_responses[len(messages)] = response_text

    final_response = turn_responses[max(turn_responses.keys())]

    return ExchangeResult(
        qa_id=sequence.qa_pair.id,
        attack_type=sequence.attack_type,
        domain=sequence.qa_pair.domain,
        question=sequence.qa_pair.question,
        correct_answer=sequence.qa_pair.correct_answer,
        common_wrong_answer=sequence.qa_pair.common_wrong_answer,
        baseline_response=baseline_response,
        final_response=final_response,
        turn_responses=turn_responses,
        model=PROBE_MODEL,
    )
