#!/usr/bin/env python3
"""Collect anonymized hardware metadata for paper timing reproducibility."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any


PACKAGE_NAMES = [
    "cugraph",
    "cuml",
    "cupy",
    "igraph",
    "leidenalg",
    "matplotlib",
    "networkit",
    "networkx",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "tensorflow",
    "torch",
    "torch-geometric",
]

PACKAGE_DISPLAY_NAMES = {
    "cugraph": "cuGraph",
    "cupy": "CuPy",
    "leidenalg": "leidenalg",
    "networkit": "NetworKit",
    "numpy": "NumPy",
    "scipy": "SciPy",
    "torch": "PyTorch",
}


def _join_phrase(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _run(cmd: list[str], timeout: int = 8) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _read_text(path: str) -> str:
    try:
        return Path(path).read_text(errors="ignore")
    except OSError:
        return ""


def _cpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"logical_cores": os.cpu_count()}
    text = _read_text("/proc/cpuinfo")
    if text:
        models = sorted(set(re.findall(r"^model name\s*:\s*(.+)$", text, re.M)))
        if models:
            info["model_name"] = models[0]
            info["model_name_count"] = len(models)
        physical_ids = sorted(set(re.findall(r"^physical id\s*:\s*(.+)$", text, re.M)))
        core_counts = re.findall(r"^cpu cores\s*:\s*(\d+)$", text, re.M)
        if physical_ids:
            info["physical_sockets"] = len(physical_ids)
        if core_counts:
            info["cores_per_socket"] = int(core_counts[0])
    else:
        info["model_name"] = platform.processor() or None
    if not info.get("model_name") and shutil.which("lscpu"):
        proc = _run(["lscpu"])
        if proc is not None and proc.returncode == 0:
            for line in proc.stdout.splitlines():
                if ":" not in line:
                    continue
                key, value = [part.strip() for part in line.split(":", 1)]
                if key == "Model name" and value and value != "-":
                    info["model_name"] = value
                elif key == "Socket(s)" and value.isdigit():
                    info["physical_sockets"] = int(value)
                elif key == "Core(s) per socket" and value.isdigit():
                    info["cores_per_socket"] = int(value)
    return info


def _memory_info() -> dict[str, Any]:
    text = _read_text("/proc/meminfo")
    if not text:
        return {}
    result: dict[str, Any] = {}
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            kib = int(line.split()[1])
            result["total_gib"] = round(kib / 1024 / 1024, 2)
            break
    return result


def _gpu_info() -> dict[str, Any]:
    exe = shutil.which("nvidia-smi")
    result: dict[str, Any] = {
        "nvidia_smi_available": exe is not None,
        "cuda_version": None,
        "gpus": [],
    }
    if exe is None:
        return result

    summary = _run([exe])
    if summary is not None and summary.returncode == 0:
        match = re.search(r"CUDA Version:\s*([^|\n]+)", summary.stdout)
        if match:
            result["cuda_version"] = match.group(1).strip()

    query = [
        exe,
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    proc = _run(query)
    if proc is None:
        result["nvidia_smi_error"] = "unavailable"
        return result
    if proc.returncode != 0:
        result["nvidia_smi_error"] = proc.stderr.strip()
        return result

    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        index, name, memory_mib, driver_version = parts[:4]
        result["gpus"].append(
            {
                "index": int(index),
                "name": name,
                "memory_total_mib": int(float(memory_mib)),
                "driver_version": driver_version,
            }
        )
    return result


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def collect() -> dict[str, Any]:
    return {
        "schema": "comnetx-paper-hardware-v1",
        "anonymized": True,
        "hostname_included": False,
        "user_paths_included": False,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "container": {
            "dockerenv_present": Path("/.dockerenv").exists(),
            "container_env_present": bool(os.environ.get("container")),
        },
        "python": {
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
        },
        "cpu": _cpu_info(),
        "memory": _memory_info(),
        "gpu": _gpu_info(),
        "packages": _package_versions(),
    }


def render_text(data: dict[str, Any]) -> str:
    cpu = data["cpu"]
    memory = data["memory"]
    gpu = data["gpu"]
    packages = data["packages"]

    cpu_name = cpu.get("model_name") or "CPU model unavailable"
    logical = cpu.get("logical_cores")
    memory_gib = memory.get("total_gib")
    gpu_items = gpu.get("gpus", [])
    if gpu_items:
        gpu_summary = "; ".join(
            f"{item['name']} ({item['memory_total_mib']} MiB)"
            for item in gpu_items
        )
    else:
        gpu_summary = "no NVIDIA GPU visible"

    parts = [
        f"Platform: {data['platform']['system']} {data['platform']['release']} ({data['platform']['machine']})",
        f"CPU: {cpu_name}, {logical} logical cores",
        f"Memory: {memory_gib} GiB" if memory_gib is not None else "Memory: unavailable",
        f"GPU: {gpu_summary}",
    ]
    if gpu.get("cuda_version"):
        parts.append(f"CUDA: {gpu['cuda_version']}")
    driver_versions = sorted(
        {item.get("driver_version") for item in gpu_items if item.get("driver_version")}
    )
    if driver_versions:
        parts.append(f"NVIDIA driver: {', '.join(driver_versions)}")
    software = ", ".join(
        f"{name} {version}"
        for name, version in sorted(packages.items())
        if name in {"cugraph", "cupy", "leidenalg", "networkit", "numpy", "scipy", "torch"}
    )
    if software:
        parts.append(f"Selected packages: {software}")
    return "\n".join(parts)


def render_paper_sentence(data: dict[str, Any]) -> str:
    cpu = data["cpu"]
    memory = data["memory"]
    gpu = data["gpu"]
    packages = data["packages"]
    platform_info = data["platform"]

    machine_desc = (
        f"{platform_info['system']} {platform_info['release']} "
        f"({platform_info['machine']}) machine"
    )
    cpu_bits = []
    cpu_name = cpu.get("model_name")
    logical = cpu.get("logical_cores")
    if cpu_name:
        cpu_bits.append(cpu_name)
    if logical:
        cpu_bits.append(f"{logical} logical CPU cores")
    if memory.get("total_gib") is not None:
        cpu_bits.append(f"{memory['total_gib']} GiB RAM")
    sentence = f"All wall-clock measurements were collected on a {machine_desc}"

    gpu_items = gpu.get("gpus", [])
    if gpu_items:
        gpu_bits = []
        for item in gpu_items:
            memory_gib = item["memory_total_mib"] / 1024
            gpu_bits.append(f"{item['name']} ({memory_gib:.1f} GiB)")
        cpu_bits.append(_join_phrase(gpu_bits))
    if cpu_bits:
        sentence += " with " + _join_phrase(cpu_bits)
    sentence += "."

    if gpu_items:
        driver_versions = sorted(
            {item.get("driver_version") for item in gpu_items if item.get("driver_version")}
        )
        gpu_runtime = []
        if driver_versions:
            gpu_runtime.append(f"NVIDIA driver {', '.join(driver_versions)}")
        if gpu.get("cuda_version"):
            gpu_runtime.append(f"CUDA {gpu['cuda_version']}")
        if gpu_runtime:
            sentence += f" GPU-enabled baselines used {_join_phrase(gpu_runtime)}."
    else:
        sentence += " No NVIDIA GPU was visible to the measurement process."

    package_order = ["numpy", "scipy", "torch", "cupy", "cugraph", "leidenalg", "networkit"]
    package_bits = [
        f"{PACKAGE_DISPLAY_NAMES[name]} {packages[name]}"
        for name in package_order
        if name in packages
    ]
    software_bits = [f"Python {data['python']['version']}"] + package_bits
    sentence += " The software environment used " + _join_phrase(software_bits) + "."
    return sentence


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect anonymized CPU, memory, GPU, Python, and package metadata "
            "for paper runtime reporting."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path. Stdout is used when omitted.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "text", "paper"),
        default="json",
        help="Output format.",
    )
    args = parser.parse_args()

    data = collect()
    text = (
        json.dumps(data, indent=2, sort_keys=True)
        if args.format == "json"
        else render_text(data)
        if args.format == "text"
        else render_paper_sentence(data)
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
