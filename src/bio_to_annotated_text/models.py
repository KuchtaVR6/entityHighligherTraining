from dataclasses import dataclass


@dataclass
class TokenRepresentation:
    token_str: str
    logits: list[float]
    label: int


@dataclass
class WordTokens:
    word_str: str
    tokens: list[TokenRepresentation]
