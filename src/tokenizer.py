from dataclasses import dataclass


@dataclass
class CharacterTokenizer:
    """
    Tiny character-level tokenizer.

    This is intentionally simple for research experiments.
    """

    stoi: dict
    itos: dict

    @classmethod
    def train(cls, text: str):
        chars = sorted(set(text))

        stoi = {
            ch: i
            for i, ch in enumerate(chars)
        }

        itos = {
            i: ch
            for ch, i in stoi.items()
        }

        return cls(
            stoi=stoi,
            itos=itos,
        )

    @property
    def vocab_size(self):
        return len(self.stoi)

    def encode(self, text: str):
        unknown = [
            ch
            for ch in text
            if ch not in self.stoi
        ]

        if unknown:
            raise ValueError(
                f"Unknown characters: {unknown[:10]}"
            )

        return [
            self.stoi[ch]
            for ch in text
        ]

    def decode(self, tokens):
        return "".join(
            self.itos[int(token)]
            for token in tokens
        )