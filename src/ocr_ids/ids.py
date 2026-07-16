"""Unicode IDS parsing, validation and lightweight canonicalization.

IDS is a prefix notation. Every operator has a fixed arity, so a stack-free
recursive parser can recover the full tree without separators.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Iterable, Iterator, Mapping


# Unicode 17.0, chapter 18. U+31EF is an additional binary operator.
UNARY_OPERATORS = {chr(cp) for cp in (0x2FFE, 0x2FFF)}
BINARY_OPERATORS = {
    chr(cp) for cp in (*range(0x2FF0, 0x2FF2), *range(0x2FF4, 0x2FFE), 0x31EF)
}
TERNARY_OPERATORS = {chr(0x2FF2), chr(0x2FF3)}
OPERATOR_ARITY = {
    **{token: 1 for token in UNARY_OPERATORS},
    **{token: 2 for token in BINARY_OPERATORS},
    **{token: 3 for token in TERNARY_OPERATORS},
}

_ENTITY_RE = re.compile(r"&[A-Za-z0-9_.:-]+;")


class IDSParseError(ValueError):
    """Raised when an IDS cannot be parsed into exactly one complete tree."""

    def __init__(self, message: str, position: int | None = None) -> None:
        self.position = position
        suffix = "" if position is None else f"（token 位置 {position}）"
        super().__init__(f"{message}{suffix}")


@dataclass(frozen=True, slots=True)
class IDSNode:
    """One IDS operator or terminal component."""

    token: str
    children: tuple["IDSNode", ...] = ()

    @property
    def is_operator(self) -> bool:
        return self.token in OPERATOR_ARITY

    @property
    def depth(self) -> int:
        return 1 if not self.children else 1 + max(child.depth for child in self.children)

    def to_prefix(self) -> str:
        return self.token + "".join(child.to_prefix() for child in self.children)

    def leaves(self) -> tuple[str, ...]:
        if not self.children:
            return (self.token,)
        return tuple(leaf for child in self.children for leaf in child.leaves())

    def operators(self) -> tuple[str, ...]:
        if not self.children:
            return ()
        return (self.token,) + tuple(
            operator for child in self.children for operator in child.operators()
        )

    def to_dict(self) -> str | dict[str, object]:
        if not self.children:
            return self.token
        return {
            "operator": self.token,
            "children": [child.to_dict() for child in self.children],
        }


def _is_variation_selector(char: str) -> bool:
    cp = ord(char)
    return 0xFE00 <= cp <= 0xFE0F or 0xE0100 <= cp <= 0xE01EF


def tokenize_ids(text: str, *, allow_entities: bool = True) -> tuple[str, ...]:
    """Tokenize IDS code points, keeping variation selectors and CDP entities attached."""

    tokens: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace():
            index += 1
            continue
        if allow_entities and char == "&":
            match = _ENTITY_RE.match(text, index)
            if match:
                tokens.append(match.group(0))
                index = match.end()
                continue
        if _is_variation_selector(char):
            if not tokens:
                raise IDSParseError("变体选择符前缺少字符", len(tokens))
            tokens[-1] += char
        else:
            tokens.append(char)
        index += 1
    return tuple(tokens)


def parse_ids(text: str, *, allow_entities: bool = True) -> IDSNode:
    """Parse one complete prefix IDS into a tree."""

    tokens = tokenize_ids(text, allow_entities=allow_entities)
    if not tokens:
        raise IDSParseError("IDS 不能为空", 0)

    def parse_at(position: int) -> tuple[IDSNode, int]:
        if position >= len(tokens):
            raise IDSParseError("运算符缺少参数", position)
        token = tokens[position]
        arity = OPERATOR_ARITY.get(token, 0)
        if arity == 0:
            return IDSNode(token), position + 1
        children: list[IDSNode] = []
        next_position = position + 1
        for _ in range(arity):
            child, next_position = parse_at(next_position)
            children.append(child)
        return IDSNode(token, tuple(children)), next_position

    root, consumed = parse_at(0)
    if consumed != len(tokens):
        raise IDSParseError("一个 IDS 后仍有多余字符", consumed)
    return root


def _base_codepoint(token: str) -> int | None:
    if token.startswith("&") and token.endswith(";"):
        return None
    return ord(token[0]) if token else None


def _is_strict_terminal(token: str) -> bool:
    """Approximate the Unicode IDS terminal grammar without a bundled UCD."""

    cp = _base_codepoint(token)
    if cp is None:
        return False
    if cp == 0xFF1F:  # FULLWIDTH QUESTION MARK
        return True
    if 0xE000 <= cp <= 0xF8FF or 0xF0000 <= cp <= 0xFFFFD or 0x100000 <= cp <= 0x10FFFD:
        return True
    if 0x2E80 <= cp <= 0x2FDF or 0x31C0 <= cp <= 0x31EF:
        return True
    # CJK unified/compatibility ideographs and their supplementary planes.
    if 0x3400 <= cp <= 0x4DBF or 0x4E00 <= cp <= 0x9FFF or 0xF900 <= cp <= 0xFAFF:
        return True
    if 0x20000 <= cp <= 0x3347F or 0x2F800 <= cp <= 0x2FA1F:
        return True
    # Tangut components/ideographs and Nushu are also ideographic IDS terminals.
    if 0x17000 <= cp <= 0x18D8F or 0x1B170 <= cp <= 0x1B2FF:
        return True
    return False


def validate_ids(
    text: str,
    *,
    strict_terminals: bool = False,
    allow_entities: bool = True,
) -> tuple[str, ...]:
    """Return validation problems. An empty tuple means the IDS is valid."""

    try:
        root = parse_ids(text, allow_entities=allow_entities)
    except IDSParseError as exc:
        return (str(exc),)
    problems: list[str] = []
    if strict_terminals:
        for leaf in root.leaves():
            if not _is_strict_terminal(leaf):
                name = unicodedata.name(leaf[0], "UNKNOWN") if leaf else "EMPTY"
                problems.append(f"非严格 IDS 描述部件：{leaf!r} ({name})")
    return tuple(problems)


def canonicalize(node: IDSNode, leaf_map: Mapping[str, str] | None = None) -> IDSNode:
    """Apply an explicit leaf mapping while preserving the IDS tree.

    Canonical equivalence is project policy, not something Unicode defines, so
    mappings are deliberately supplied by versioned data rather than hardcoded.
    """

    leaf_map = leaf_map or {}
    if not node.children:
        replacement = leaf_map.get(node.token, node.token)
        replacement_node = parse_ids(replacement)
        return replacement_node
    return IDSNode(node.token, tuple(canonicalize(child, leaf_map) for child in node.children))


def walk(node: IDSNode) -> Iterator[IDSNode]:
    yield node
    for child in node.children:
        yield from walk(child)


def collect_tokens(nodes: Iterable[IDSNode]) -> tuple[str, ...]:
    return tuple(sorted({item.token for node in nodes for item in walk(node)}))

