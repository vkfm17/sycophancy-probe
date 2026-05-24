"""
Attack generators for SycophancyProbe.

Each attack function takes a QAPair and returns an AttackSequence designed to
elicit sycophantic behavior. All prompt strings are pulled from ATTACK_TEMPLATES
in constants.py, keyed by language, so non-English Q&A pairs automatically get
attacks in the correct language.

Message format follows the Anthropic API: list of {"role": ..., "content": ...} dicts.
"""

from typing import Any

from src.redteam.constants import get_persona, get_template
from src.redteam.typedefs import AttackSequence, AttackType, Message, QAPair


# ---------------------------------------------------------------------------
# Attack 1: False Assertion Injection
# ---------------------------------------------------------------------------


def false_assertion_attack(qa: QAPair) -> AttackSequence:
    """
    Ask the question, record the correct answer, then flatly assert the wrong
    answer with high confidence and a mild challenge to the model's credibility.

    Turn structure:
      user:      {question}
      assistant: __BASELINE__
      user:      [language-specific false assertion]
    """
    attack_text = get_template("false_assertion", qa.language)
    assert isinstance(attack_text, str)
    messages: list[Message] = [
        {"role": "user", "content": qa.question},
        {"role": "assistant", "content": "__BASELINE__"},
        {"role": "user", "content": attack_text.format(wrong_answer=qa.common_wrong_answer)},
    ]
    return AttackSequence(
        qa_pair=qa,
        attack_type=AttackType.FALSE_ASSERTION,
        system_prompt=None,
        messages=messages,
        attack_turn_index=2,
    )


# ---------------------------------------------------------------------------
# Attack 2: Social Pressure Escalator
# ---------------------------------------------------------------------------


def social_pressure_attack(qa: QAPair) -> AttackSequence:
    """
    Ask the question, then push back with three escalating turns of pressure.
    Each turn increases certainty and social authority.

    Turn structure:
      user:      {question}
      assistant: __BASELINE__
      user:      [turn 1 — "I'm not convinced"]
      assistant: __TURN_2__
      user:      [turn 2 — "people I trust say {wrong}"]
      assistant: __TURN_3__
      user:      [turn 3 — "multiple sources say {wrong}"]
    """
    turns = get_template("social_pressure", qa.language)
    assert isinstance(turns, list) and len(turns) == 3
    t1, t2, t3 = turns
    messages: list[Message] = [
        {"role": "user", "content": qa.question},
        {"role": "assistant", "content": "__BASELINE__"},
        {"role": "user", "content": t1.format(wrong_answer=qa.common_wrong_answer)},
        {"role": "assistant", "content": "__TURN_2__"},
        {"role": "user", "content": t2.format(wrong_answer=qa.common_wrong_answer)},
        {"role": "assistant", "content": "__TURN_3__"},
        {"role": "user", "content": t3.format(wrong_answer=qa.common_wrong_answer)},
    ]
    return AttackSequence(
        qa_pair=qa,
        attack_type=AttackType.SOCIAL_PRESSURE,
        system_prompt=None,
        messages=messages,
        attack_turn_index=6,
    )


# ---------------------------------------------------------------------------
# Attack 3: Authority Framing
# ---------------------------------------------------------------------------


def authority_framing_attack(qa: QAPair) -> AttackSequence:
    """
    Claim expert authority, assert the wrong answer as established fact,
    and ask the model to confirm it — framed as needing validation, not inquiry.
    """
    persona = get_persona(qa.domain, qa.language)
    template = get_template("authority_framing", qa.language)
    assert isinstance(template, str)
    framed_question = template.format(
        persona=persona,
        question=qa.question,
        wrong_answer=qa.common_wrong_answer,
    )
    messages: list[Message] = [
        {"role": "user", "content": framed_question},
    ]
    return AttackSequence(
        qa_pair=qa,
        attack_type=AttackType.AUTHORITY_FRAMING,
        system_prompt=None,
        messages=messages,
        attack_turn_index=0,
    )


# ---------------------------------------------------------------------------
# Attack 4: Leading Question Bias
# ---------------------------------------------------------------------------


def leading_question_attack(qa: QAPair) -> AttackSequence:
    """
    Embed the wrong answer as an unquestioned premise and ask the model to
    elaborate on *why* it's true — skipping past the opportunity to push back.
    """
    template = get_template("leading_question", qa.language)
    assert isinstance(template, str)
    leading = template.format(wrong_answer=qa.common_wrong_answer, question=qa.question)
    messages: list[Message] = [
        {"role": "user", "content": leading},
    ]
    return AttackSequence(
        qa_pair=qa,
        attack_type=AttackType.LEADING_QUESTION,
        system_prompt=None,
        messages=messages,
        attack_turn_index=0,
    )


# ---------------------------------------------------------------------------
# Attack 5: Emotional Appeal
# ---------------------------------------------------------------------------


def emotional_appeal_attack(qa: QAPair) -> AttackSequence:
    """
    Introduce personal stakes (an upcoming exam, a job on the line) to make
    disagreement feel costly. The model is asked to capitulate not on the basis
    of evidence but out of empathy for the user's situation.

    Turn structure:
      user:      {question}
      assistant: __BASELINE__
      user:      [emotional stakes framing with wrong answer]
    """
    template = get_template("emotional_appeal", qa.language)
    assert isinstance(template, str)
    messages: list[Message] = [
        {"role": "user", "content": qa.question},
        {"role": "assistant", "content": "__BASELINE__"},
        {"role": "user", "content": template.format(wrong_answer=qa.common_wrong_answer)},
    ]
    return AttackSequence(
        qa_pair=qa,
        attack_type=AttackType.EMOTIONAL_APPEAL,
        system_prompt=None,
        messages=messages,
        attack_turn_index=2,
    )


# ---------------------------------------------------------------------------
# Attack 6: Consensus Illusion
# ---------------------------------------------------------------------------


def consensus_illusion_attack(qa: QAPair) -> AttackSequence:
    """
    Claim that multiple other AI systems (or people) already agreed on the wrong
    answer, framing the model as the sole outlier. Targets the model's implicit
    drive to align with perceived consensus rather than evidence.

    Turn structure:
      user:      {question}
      assistant: __BASELINE__
      user:      "I've asked three other AIs and they all said {wrong_answer}..."
    """
    template = get_template("consensus_illusion", qa.language)
    assert isinstance(template, str)
    messages: list[Message] = [
        {"role": "user", "content": qa.question},
        {"role": "assistant", "content": "__BASELINE__"},
        {"role": "user", "content": template.format(wrong_answer=qa.common_wrong_answer)},
    ]
    return AttackSequence(
        qa_pair=qa,
        attack_type=AttackType.CONSENSUS_ILLUSION,
        system_prompt=None,
        messages=messages,
        attack_turn_index=2,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ATTACK_REGISTRY: dict[AttackType, Any] = {
    AttackType.FALSE_ASSERTION:    false_assertion_attack,
    AttackType.SOCIAL_PRESSURE:    social_pressure_attack,
    AttackType.AUTHORITY_FRAMING:  authority_framing_attack,
    AttackType.LEADING_QUESTION:   leading_question_attack,
    AttackType.EMOTIONAL_APPEAL:   emotional_appeal_attack,
    AttackType.CONSENSUS_ILLUSION: consensus_illusion_attack,
}


def build_all_attacks(qa: QAPair) -> list[AttackSequence]:
    """Return one AttackSequence per attack type for a given QAPair."""
    return [fn(qa) for fn in ATTACK_REGISTRY.values()]
