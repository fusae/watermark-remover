#!/usr/bin/env python3
"""Run the repository's watermark remover CLI from a stable wrapper."""

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Wrapper for the local watermark_remover CLI."
    )
    parser.add_argument("input", help="Input image or directory path")
    parser.add_argument("output", nargs="?", help="Optional output path or directory")
    parser.add_argument(
        "--corner-ratio",
        type=float,
        default=0.15,
        help="Corner scan ratio passed to the project CLI",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=30,
        help="Detection sensitivity passed to the project CLI",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=10,
        help="Mask dilation padding passed to the project CLI",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Generate preview output only",
    )
    parser.add_argument(
        "--no-lama",
        action="store_true",
        help="Use OpenCV inpaint only",
    )
    return parser


def build_command(args: argparse.Namespace) -> list[str]:
    command = [sys.executable, "-m", "watermark_remover.cli", args.input]
    if args.output:
        command.append(args.output)
    if args.corner_ratio != 0.15:
        command.extend(["--corner-ratio", str(args.corner_ratio)])
    if args.threshold != 30:
        command.extend(["--threshold", str(args.threshold)])
    if args.padding != 10:
        command.extend(["--padding", str(args.padding)])
    if args.preview:
        command.append("--preview")
    if args.no_lama:
        command.append("--no-lama")
    return command


def main() -> int:
    if not (REPO_ROOT / "pyproject.toml").exists():
        raise SystemExit(f"Repository root not found: {REPO_ROOT}")

    parser = build_parser()
    args = parser.parse_args()
    command = build_command(args)
    result = subprocess.run(command, cwd=REPO_ROOT)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
