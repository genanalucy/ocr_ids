"""Command line interface for core IDS operations."""

from __future__ import annotations

import argparse
import json

from .ids import parse_ids, validate_ids


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ocr-ids")
    commands = parser.add_subparsers(dest="command", required=True)

    parse_command = commands.add_parser("parse", help="将 IDS 转换为结构树")
    parse_command.add_argument("ids")

    validate_command = commands.add_parser("validate", help="校验 IDS")
    validate_command.add_argument("ids")
    validate_command.add_argument("--strict-terminals", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "parse":
        node = parse_ids(args.ids)
        print(
            json.dumps(
                {
                    "ids": node.to_prefix(),
                    "tree": node.to_dict(),
                    "depth": node.depth,
                    "leaves": node.leaves(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "validate":
        problems = validate_ids(args.ids, strict_terminals=args.strict_terminals)
        if problems:
            for problem in problems:
                print(problem)
            return 1
        print("valid")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

