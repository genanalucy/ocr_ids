"""The deliberately narrow, flat IDS scope for phase-one production data."""

from __future__ import annotations

from .ids import parse_ids


FLAT_STRUCTURE_OPERATORS = {
    "⿰": "left_right",
    "⿱": "top_bottom",
    "⿲": "left_middle_right",
    "⿳": "top_middle_bottom",
}


def flat_structure_kind(ids: str) -> str | None:
    """Classify an IDS in the first-release scope, otherwise return ``None``.

    A supported sample has exactly one spatial operator at its root and only
    terminal components below it. This intentionally excludes enclosure and
    every nested composition, even when its inner operators are supported.
    """

    root = parse_ids(ids)
    kind = FLAT_STRUCTURE_OPERATORS.get(root.token)
    if kind is None or any(child.children for child in root.children):
        return None
    return kind
