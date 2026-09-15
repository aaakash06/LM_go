from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CharacterTokenizer:
    """
    Simple character-level tokenizer.

    Every unique character receives an integer ID.
    """

    stoi: dict[str, int]
    itos: dict[int, str]

    @classmethod
    def train(cls, text: str) -> "CharacterTokenizer":
        characters = sorted(set(text))

        stoi = {
            character: index
            for index, character in enumerate(characters)
        }

        itos = {
            index: character
            for character, index in stoi.items()
        }

        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        try:
            return [self.stoi[c] for c in text]
        except KeyError as exc:
            raise ValueError(
                f"Unknown character: {exc.args[0]!r}"
            ) from exc

    def decode(self, tokens: list[int]) -> str:
        return "".join(
            self.itos[token]
            for token in tokens
        )