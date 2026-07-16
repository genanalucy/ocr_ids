"""Project-level canonical corpus policies."""

from __future__ import annotations


HAN_TARGET_RANGES = (
    (0x3400, 0x4DBF),  # CJK Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0x20000, 0x3347F),  # Extensions B–J in the current project scope
    (0x2F800, 0x2FA1F),  # Compatibility Ideographs Supplement
)


def is_han_target(character: str | None) -> bool:
    """Whether a sample target is an encoded CJK Han ideograph."""

    if not character or len(character) != 1:
        return False
    codepoint = ord(character)
    return any(start <= codepoint <= end for start, end in HAN_TARGET_RANGES)

