"""IDS token vocabulary."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

from .ids import parse_ids, walk


SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")


@dataclass(frozen=True, slots=True)
class Vocabulary:
    tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.tokens) != len(set(self.tokens)):
            raise ValueError("词表包含重复 token")
        if self.tokens[: len(SPECIAL_TOKENS)] != SPECIAL_TOKENS:
            raise ValueError("词表必须以固定特殊 token 开头")

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    def encode(self, ids: str, *, add_special_tokens: bool = True) -> list[int]:
        mapping = self.token_to_id
        sequence = [item.token for item in walk(parse_ids(ids))]
        encoded = [mapping.get(token, mapping["<unk>"]) for token in sequence]
        if add_special_tokens:
            encoded = [mapping["<bos>"], *encoded, mapping["<eos>"]]
        return encoded

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"tokens": self.tokens}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(tuple(data["tokens"]))


def build_vocabulary(ids_values: Iterable[str]) -> Vocabulary:
    tokens = {item.token for ids in ids_values for item in walk(parse_ids(ids))}
    return Vocabulary(SPECIAL_TOKENS + tuple(sorted(tokens)))

