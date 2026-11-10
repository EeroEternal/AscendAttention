"""``python -m ascend_attention`` — produce the two artefacts a platform reads.

* default: regenerate the committed ``manifest.json`` (the *usable* subset; CI diffs it);
* ``--capabilities``: print the **full inventory** as JSON, including implementations that are disabled
  or lack a measured ratio.

Why two artefacts instead of one: the manifest answers "what may I call today?", which is what a platform
wants at start-up, while the inventory answers "what exists, and why is it not usable?" — the question a
platform asks when it has to *reject* a shape early (``graph_capturable`` on an op with no enabled
implementation is exactly such a case, and it cannot appear in a manifest that only lists usable ops).
Both come from the same ``ImplSpec`` data, so they cannot disagree.
"""

from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

from . import capabilities as capabilities_fn
from .manifest import write


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ascend_attention", description=__doc__)
    parser.add_argument(
        "--capabilities",
        action="store_true",
        help="print the full implementation inventory (including disabled impls) as JSON",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="no indentation (smaller, still diffable per key)",
    )
    args = parser.parse_args(argv)

    if args.capabilities:
        text = json.dumps(capabilities_fn(), indent=None if args.compact else 2, sort_keys=False)
        print(text)
        return 0

    target = write()
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
