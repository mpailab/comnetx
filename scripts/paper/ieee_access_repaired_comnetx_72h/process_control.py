"""Bounded process-group supervision for measurement commands."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Callable


POLL_SECONDS = 0.05
INTERRUPT_GRACE_SECONDS = 5.0
KILL_SETTLE_SECONDS = 0.2


class LauncherSignalInterrupt(RuntimeError):
    """Raised when TERM or HUP asks the Python launcher to clean up and stop."""


def install_termination_signal_handlers() -> None:
    """Turn termination signals into exceptions so cleanup blocks always run."""

    def interrupt(signum: int, _frame: object) -> None:
        raise LauncherSignalInterrupt(f"launcher received signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, interrupt)


def run_streaming_process(
    command: list[str],
    environment: dict[str, str],
    log_path: Path,
    *,
    cwd: Path,
    deadline: float | None = None,
    deadline_kill_grace_seconds: float = 60.0,
    on_start: Callable[[int], None] | None = None,
) -> tuple[int, bool]:
    """Stream one command and leave no live member of its private process group."""

    if deadline_kill_grace_seconds < 0:
        raise ValueError("deadline kill grace must be non-negative")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        process_group = process.pid
        cancel_watchdog = threading.Event()
        deadline_signal_sent = threading.Event()
        deadline_cleanup_complete = threading.Event()

        def group_exists() -> bool:
            process.poll()
            try:
                os.killpg(process_group, 0)
            except ProcessLookupError:
                return False
            except PermissionError:
                return True
            return True

        def signal_group(signum: int) -> bool:
            try:
                os.killpg(process_group, signum)
            except ProcessLookupError:
                return False
            return True

        def wait_for_group_exit(seconds: float) -> bool:
            end = time.monotonic() + max(0.0, seconds)
            while group_exists():
                remaining = end - time.monotonic()
                if remaining <= 0:
                    return False
                time.sleep(min(POLL_SECONDS, remaining))
            return True

        def terminate_group(grace_seconds: float) -> None:
            if not signal_group(signal.SIGTERM):
                return
            if wait_for_group_exit(grace_seconds):
                return
            signal_group(signal.SIGKILL)
            wait_for_group_exit(1.0)

        def deadline_watchdog() -> None:
            assert deadline is not None
            delay = max(0.0, deadline - time.monotonic())
            if cancel_watchdog.wait(delay):
                return
            try:
                parent_was_running = process.poll() is None
                if signal_group(signal.SIGTERM):
                    if parent_was_running:
                        deadline_signal_sent.set()
                    if not wait_for_group_exit(deadline_kill_grace_seconds):
                        signal_group(signal.SIGKILL)
                        wait_for_group_exit(KILL_SETTLE_SECONDS)
            finally:
                deadline_cleanup_complete.set()

        watchdog = None
        if deadline is not None:
            watchdog = threading.Thread(
                target=deadline_watchdog,
                name=f"measurement-deadline-{process.pid}",
                daemon=True,
            )
            watchdog.start()

        return_code: int | None = None
        try:
            if on_start is not None:
                on_start(process.pid)
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            return_code = process.wait()
        finally:
            interrupted = sys.exc_info()[0] is not None
            cancel_watchdog.set()
            if interrupted:
                terminate_group(INTERRUPT_GRACE_SECONDS)
            if watchdog is not None:
                watchdog.join()
            if not deadline_cleanup_complete.is_set() and group_exists():
                terminate_group(INTERRUPT_GRACE_SECONDS)
            if process.poll() is None:
                process.wait()

        assert return_code is not None
        stopped_by_deadline = deadline_signal_sent.is_set()
        return return_code, stopped_by_deadline
