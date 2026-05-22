"""Shared type definitions for the redteam layer."""

from typing import TypedDict


class Message(TypedDict):
    role: str
    content: str
