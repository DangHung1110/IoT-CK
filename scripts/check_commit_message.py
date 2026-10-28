"""Validate Conventional Commit messages for the commit-msg Git hook."""

from __future__ import annotations

import re
import sys
from pathlib import Path

PATTERN = re.compile(
    r"^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)"
    r"(?:\([a-z0-9][a-z0-9-]*\))?!?: .{1,72}$"
)


def main() -> int:
    message_file = Path(sys.argv[1])
    first_line = message_file.read_text(encoding="utf-8").splitlines()[0]
    if first_line.startswith(("Merge ", "Revert ")):
        return 0
    if PATTERN.fullmatch(first_line):
        return 0
    print(
        "Commit message must use Conventional Commits, for example: "
        "chore(env): prepare Python workspace",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
