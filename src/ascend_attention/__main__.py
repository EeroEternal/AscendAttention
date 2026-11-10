"""``python -m ascend_attention`` — regenerate the committed manifest (CI diffs the result)."""

from __future__ import annotations

from .manifest import write


def main() -> int:
    target = write()
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
