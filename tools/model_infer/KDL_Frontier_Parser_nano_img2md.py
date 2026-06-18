#!/usr/bin/env python3
"""Run KDL-Frontier-Parser-nano over page images and write OmniDocBench Markdown."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    from tools.omnidocbench_batch_infer import main as batch_main

    argv = sys.argv[1:]
    if "--engine" not in argv:
        argv = ["--engine", "kdl_frontier_nano", *argv]
    original_argv = sys.argv
    try:
        sys.argv = [str(Path(__file__)), *argv]
        batch_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
