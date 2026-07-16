"""OCR-IDS core package."""

from .ids import IDSNode, IDSParseError, canonicalize, parse_ids, validate_ids
from .canonical import is_han_target

__all__ = [
    "IDSNode",
    "IDSParseError",
    "canonicalize",
    "parse_ids",
    "validate_ids",
    "is_han_target",
]

__version__ = "0.1.0"
