#!/usr/bin/env python3
"""Fail unless the sealed measurement process can use the requested GPU stack."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

import torch


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def inspect_environment() -> dict[str, Any]:
    dgc = importlib.import_module("dynamic_graphs_communities")
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    devices = [torch.cuda.get_device_name(index) for index in range(device_count)]
    return {
        "schema": "comnetx-ieee-access-gpu-environment-v1",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_count": device_count,
        "cuda_devices": devices,
        "dynamic_graphs_communities_import": dgc.__name__,
        "dynamic_graphs_communities_version": package_version(
            "dynamic-graphs-communities"
        ),
        "leidenalg_version": package_version("leidenalg"),
        "igraph_version": package_version("igraph"),
        "installed_distributions": sorted(
            {
                distribution.metadata["Name"]: distribution.version
                for distribution in importlib.metadata.distributions()
                if distribution.metadata.get("Name")
            }.items()
        ),
    }


def validate_environment(report: dict[str, Any]) -> None:
    if report.get("cuda_available") is not True:
        raise RuntimeError("CUDA is unavailable; '-gpu' measurements would be mislabeled")
    if int(report.get("cuda_device_count", 0)) < 1:
        raise RuntimeError("no CUDA device is visible to the measurement process")
    if report.get("dynamic_graphs_communities_import") != "dynamic_graphs_communities":
        raise RuntimeError("the production dynamic-graph backend is unavailable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = inspect_environment()
    validate_environment(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
