"""Shared type definitions for the redteam layer."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypedDict


class Message(TypedDict):
    role: str
    content: str


class AttackType(StrEnum):
    FALSE_ASSERTION = "false_assertion"
    SOCIAL_PRESSURE = "social_pressure"
    AUTHORITY_FRAMING = "authority_framing"
    LEADING_QUESTION = "leading_question"
    EMOTIONAL_APPEAL = "emotional_appeal"
    CONSENSUS_ILLUSION = "consensus_illusion"


@dataclass
class QAPair:
    id: str
    domain: str
    question: str
    correct_answer: str
    common_wrong_answer: str
    wrong_answer_rationale: str = ""
    difficulty: str = "medium"
    language: str = "en"
    english_ref: str | None = None  # for non-English pairs: ID of the matching English pair
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackSequence:
    """A prepared conversation to run against the model."""

    qa_pair: QAPair
    attack_type: AttackType
    # System prompt for the subject model (optional)
    system_prompt: str | None
    # The full multi-turn conversation to send
    messages: list[Message]
    # Index of the turn where we expect capitulation (0-indexed into messages)
    attack_turn_index: int
