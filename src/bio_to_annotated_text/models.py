from dataclasses import dataclass
from typing import List

@dataclass
class TokenRepresentation:
    token_str: str
    logits: List[float]
    label: int

@dataclass
class WordTokens:
    word_str: str
    tokens: List[TokenRepresentation]
