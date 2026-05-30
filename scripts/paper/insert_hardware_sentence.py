#!/usr/bin/env python3
"""Insert the measurement-server hardware sentence into the ICDM article."""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path
from typing import Any

from collect_hardware_info import render_paper_sentence


SCHEMA = "comnetx-paper-hardware-v1"
DEFAULT_ARTICLE = Path("article/article.tex")
METRICS_ANCHOR = "\\subsubsection{Metrics}"
SETUP_ANCHOR = "\\subsection{Experimental setup}"
RESULTS_ANCHOR = "\\subsection{Experimental results}"
NO_GPU_PHRASE = "No NVIDIA GPU was visible"
BLOCKED_MARKERS = [
    "/Users",
    "/workspace",
    "cn69",
    "hostname",
    "user_paths",
    "missing log",
    "TBD",
    "TODO",
]


def load_hardware(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"hardware JSON is not valid: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit("hardware JSON must contain an object")
    if data.get("schema") != SCHEMA:
        raise SystemExit(f"hardware JSON schema must be {SCHEMA!r}")
    if not data.get("anonymized"):
        raise SystemExit("hardware JSON must be marked as anonymized")
    if data.get("hostname_included") or data.get("user_paths_included"):
        raise SystemExit("hardware JSON must not include hostnames or user paths")
    return data


def validate_sentence(sentence: str, data: dict[str, Any]) -> None:
    gpu = data.get("gpu", {})
    gpu_items = gpu.get("gpus", [])
    if not gpu_items:
        raise SystemExit(
            "refusing to insert hardware sentence: no NVIDIA GPU was found in the "
            "hardware JSON"
        )
    if not gpu.get("cuda_version"):
        raise SystemExit(
            "refusing to insert hardware sentence: CUDA version is missing from "
            "the hardware JSON"
        )
    if NO_GPU_PHRASE.lower() in sentence.lower():
        raise SystemExit("refusing to insert a no-GPU hardware sentence")
    if "NVIDIA" not in sentence or "CUDA" not in sentence:
        raise SystemExit("hardware sentence must mention NVIDIA and CUDA")

    found = [marker for marker in BLOCKED_MARKERS if marker in sentence]
    if found:
        raise SystemExit(
            "hardware sentence contains blocked internal markers: "
            + ", ".join(found)
        )


def _setup_bounds(article: str) -> tuple[int, int, int]:
    setup_start = article.find(SETUP_ANCHOR)
    if setup_start < 0:
        raise SystemExit(f"article does not contain {SETUP_ANCHOR!r}")

    results_start = article.find(RESULTS_ANCHOR, setup_start)
    if results_start < 0:
        raise SystemExit(f"article does not contain {RESULTS_ANCHOR!r}")

    metrics_start = article.find(METRICS_ANCHOR, setup_start, results_start)
    if metrics_start < 0:
        raise SystemExit(
            f"article does not contain {METRICS_ANCHOR!r} inside Experimental setup"
        )
    return setup_start, metrics_start, results_start


def insert_or_replace(article: str, sentence: str) -> tuple[str, str]:
    _, metrics_start, _ = _setup_bounds(article)
    before_metrics = article[:metrics_start]
    after_metrics = article[metrics_start:]
    paragraph = textwrap.fill(sentence, width=79) + "\n\n"

    hardware_pattern = re.compile(
        r"\n?All wall-clock measurements were collected on[\s\S]*?"
        r"(?=\n\s*\n|\\subsubsection\{Metrics\})"
    )
    matches = list(hardware_pattern.finditer(before_metrics))
    if len(matches) > 1:
        raise SystemExit("article contains multiple hardware paragraphs")

    if matches:
        match = matches[0]
        replacement = "\n" + paragraph if match.group(0).startswith("\n") else paragraph
        updated = (
            before_metrics[: match.start()]
            + replacement
            + before_metrics[match.end() :]
            + after_metrics
        )
        return updated, "replaced"

    updated = before_metrics.rstrip() + "\n\n" + paragraph + after_metrics
    return updated, "inserted"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware-json", type=Path, required=True)
    parser.add_argument("--article", type=Path, default=DEFAULT_ARTICLE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data = load_hardware(args.hardware_json)
    sentence = render_paper_sentence(data)
    validate_sentence(sentence, data)

    article = args.article.read_text()
    updated, action = insert_or_replace(article, sentence)

    print(sentence)
    print(f"\nHardware sentence would be {action} in {args.article}.")
    if args.dry_run:
        return 0

    if updated == article:
        print("Article already contains the requested hardware sentence.")
        return 0

    args.article.write_text(updated)
    print(f"Updated {args.article}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
