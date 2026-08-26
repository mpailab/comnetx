#!/usr/bin/env python3
"""Maintain the append-only measurement windows for the IEEE Access campaign."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence


SCHEMA = "comnetx-ieee-access-launch-budget-v2"
INITIAL_WINDOW_HOURS = 24.0
MAX_TOTAL_HOURS = 72.0


class BudgetError(RuntimeError):
    """Raised when a saved measurement budget is invalid or unsafe to change."""


def _finite_positive_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BudgetError(f"{field} must be a positive finite number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise BudgetError(f"{field} must be a positive finite number")
    return number


def _hours_to_seconds(hours: float, *, field: str) -> int:
    seconds = int(round(hours * 3600.0))
    if seconds <= 0:
        raise BudgetError(f"{field} is too small to grant a measurement window")
    return seconds


def _paths_config_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise BudgetError(f"cannot read paths config {path}: {exc}") from exc


def _expected_identity(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "git_sha": args.git_sha,
        "repaired_campaign_id": args.repaired_campaign_id,
        "ld_campaign_id": args.ld_campaign_id,
        "paths_config_sha256": _paths_config_sha256(args.paths_config),
        "max_total_hours": MAX_TOTAL_HOURS,
        "initial_window_hours": INITIAL_WINDOW_HOURS,
    }


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BudgetError(f"measurement budget does not exist: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BudgetError(f"cannot read measurement budget {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BudgetError("measurement budget must be a JSON object")
    return payload


def _validate(payload: dict[str, Any], expected: dict[str, Any]) -> list[dict[str, Any]]:
    for field, value in expected.items():
        if payload.get(field) != value:
            raise BudgetError(f"saved measurement budget mismatch for {field}")

    windows = payload.get("windows")
    if not isinstance(windows, list) or not windows:
        raise BudgetError("measurement budget must contain at least one window")

    total_seconds = 0
    previous_deadline: int | None = None
    validated: list[dict[str, Any]] = []
    for index, window in enumerate(windows, start=1):
        if not isinstance(window, dict):
            raise BudgetError(f"measurement window {index} must be a JSON object")
        expected_id = f"window-{index:03d}"
        if window.get("id") != expected_id:
            raise BudgetError(
                f"measurement window {index} has id {window.get('id')!r}; "
                f"expected {expected_id!r}"
            )
        granted_hours = _finite_positive_number(
            window.get("granted_hours"), field=f"{expected_id}.granted_hours"
        )
        granted_seconds = _hours_to_seconds(
            granted_hours, field=f"{expected_id}.granted_hours"
        )
        if granted_hours != granted_seconds / 3600.0:
            raise BudgetError(
                f"{expected_id}.granted_hours does not encode whole seconds"
            )
        started = window.get("started_at_epoch")
        deadline = window.get("deadline_epoch")
        if (
            isinstance(started, bool)
            or not isinstance(started, int)
            or isinstance(deadline, bool)
            or not isinstance(deadline, int)
        ):
            raise BudgetError(f"{expected_id} has malformed timestamps")
        if deadline != started + granted_seconds:
            raise BudgetError(f"{expected_id} deadline is inconsistent with its grant")
        if previous_deadline is not None and started < previous_deadline:
            raise BudgetError(f"{expected_id} overlaps the preceding measurement window")
        previous_deadline = deadline
        total_seconds += granted_seconds
        validated.append(window)

    if float(validated[0]["granted_hours"]) != INITIAL_WINDOW_HOURS:
        raise BudgetError("window-001 must be the fixed 24-hour initial window")
    maximum_seconds = _hours_to_seconds(MAX_TOTAL_HOURS, field="max_total_hours")
    if total_seconds > maximum_seconds:
        raise BudgetError("saved measurement windows exceed the 72-hour cap")
    return validated


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def _exclusive_state_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield


def _new_window(window_number: int, granted_hours: float, now_epoch: int) -> dict[str, Any]:
    granted_seconds = _hours_to_seconds(granted_hours, field="granted_hours")
    return {
        "id": f"window-{window_number:03d}",
        "granted_hours": granted_seconds / 3600.0,
        "started_at_epoch": now_epoch,
        "deadline_epoch": now_epoch + granted_seconds,
    }


def initialize(
    path: Path,
    expected: dict[str, Any],
    initial_hours: float,
    *,
    now_epoch: int | None = None,
) -> dict[str, Any]:
    initial_hours = _finite_positive_number(initial_hours, field="initial window hours")
    if initial_hours != INITIAL_WINDOW_HOURS:
        raise BudgetError("the initial measurement window is fixed at 24 hours")
    with _exclusive_state_lock(path):
        if path.exists():
            payload = _load(path)
            _validate(payload, expected)
            return payload
        started = int(time.time()) if now_epoch is None else int(now_epoch)
        payload = {
            **expected,
            "windows": [_new_window(1, INITIAL_WINDOW_HOURS, started)],
        }
        _atomic_write(path, payload)
        return payload


def _running_metadata(metadata_roots: Sequence[Path]) -> list[Path]:
    running: list[Path] = []
    for root in metadata_roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("**/metadata.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise BudgetError(f"cannot audit attempt metadata {path}: {exc}") from exc
            if not isinstance(payload, dict):
                raise BudgetError(f"attempt metadata is not a JSON object: {path}")
            if payload.get("status") == "running":
                running.append(path)
    return running


def extend(
    path: Path,
    expected: dict[str, Any],
    extension_hours: float,
    metadata_roots: Sequence[Path],
    *,
    now_epoch: int | None = None,
) -> dict[str, Any]:
    extension_hours = _finite_positive_number(
        extension_hours, field="CAMPAIGN_EXTENSION_HOURS"
    )
    extension_seconds = _hours_to_seconds(
        extension_hours, field="CAMPAIGN_EXTENSION_HOURS"
    )
    with _exclusive_state_lock(path):
        payload = _load(path)
        windows = _validate(payload, expected)
        current_time = int(time.time()) if now_epoch is None else int(now_epoch)
        previous_deadline = int(windows[-1]["deadline_epoch"])
        if current_time < previous_deadline:
            raise BudgetError(
                "cannot append a measurement window before the current window expires"
            )
        running = _running_metadata(metadata_roots)
        if running:
            listing = ", ".join(str(item) for item in running[:5])
            if len(running) > 5:
                listing += f", and {len(running) - 5} more"
            raise BudgetError(
                "cannot append a measurement window while attempt metadata is "
                f"status=running: {listing}"
            )
        granted_seconds = sum(
            _hours_to_seconds(
                float(window["granted_hours"]),
                field=f"{window['id']}.granted_hours",
            )
            for window in windows
        )
        maximum_seconds = _hours_to_seconds(MAX_TOTAL_HOURS, field="max_total_hours")
        if granted_seconds + extension_seconds > maximum_seconds:
            remaining_hours = (maximum_seconds - granted_seconds) / 3600.0
            raise BudgetError(
                "extension exceeds the 72-hour cap; "
                f"at most {remaining_hours:g} hours remain grantable"
            )
        # The identity fields and all existing windows are preserved byte-for-value;
        # an extension can only append this newly registered, non-overlapping window.
        updated = dict(payload)
        updated["windows"] = [
            *windows,
            _new_window(len(windows) + 1, extension_seconds / 3600.0, current_time),
        ]
        _atomic_write(path, updated)
        return updated


def _current_window(payload: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    return _validate(payload, expected)[-1]


def _add_identity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-file", required=True, type=Path)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--repaired-campaign-id", required=True)
    parser.add_argument("--ld-campaign-id", required=True)
    parser.add_argument("--paths-config", required=True, type=Path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "initialize",
        "extend",
        "hours-left",
        "deadline",
        "window-id",
        "granted-hours",
    ):
        child = subparsers.add_parser(command)
        _add_identity_arguments(child)
        if command == "initialize":
            child.add_argument(
                "--initial-hours", type=float, default=INITIAL_WINDOW_HOURS
            )
        elif command == "extend":
            child.add_argument("--extension-hours", required=True, type=float)
            child.add_argument(
                "--metadata-root", required=True, action="append", type=Path
            )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        expected = _expected_identity(args)
        if args.command == "initialize":
            payload = initialize(
                args.state_file, expected, args.initial_hours
            )
            window = _current_window(payload, expected)
            print(
                f"{window['id']} {float(window['granted_hours']):g} "
                f"{window['deadline_epoch']}"
            )
            return 0
        if args.command == "extend":
            payload = extend(
                args.state_file,
                expected,
                args.extension_hours,
                args.metadata_root,
            )
            window = _current_window(payload, expected)
            print(
                f"{window['id']} {float(window['granted_hours']):g} "
                f"{window['deadline_epoch']}"
            )
            return 0

        payload = _load(args.state_file)
        window = _current_window(payload, expected)
        if args.command == "hours-left":
            remaining = max(
                0.0, (float(window["deadline_epoch"]) - time.time()) / 3600.0
            )
            if not math.isfinite(remaining):
                raise BudgetError("invalid remaining-time calculation")
            print(f"{remaining:.6f}")
        elif args.command == "deadline":
            print(window["deadline_epoch"])
        elif args.command == "window-id":
            print(window["id"])
        elif args.command == "granted-hours":
            print(f"{float(window['granted_hours']):g}")
        else:  # pragma: no cover - argparse constrains the command.
            raise BudgetError(f"unsupported command: {args.command}")
        return 0
    except BudgetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
