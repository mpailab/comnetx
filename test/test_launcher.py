import subprocess, sys
import os
import pytest
import re
from pathlib import Path
import signal, time
import shutil 

TEST_PATH = Path(__file__).resolve().parent
PROJECT_PATH = TEST_PATH.parent

for p in (PROJECT_PATH, PROJECT_PATH / "src", TEST_PATH):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from launcher import dynamic_launch
from testutils import with_timeout, TimeoutException

@pytest.mark.short
def test_dynamic_prgpt():
    for method in ["prgpt:locale", "prgpt:infomap"]:
        for mode in ["smart", "naive", "raw"]:
            dynamic_launch("wiki_talk_ht", 10, method, mode = mode)

@pytest.mark.short  
def test_dynamic_leidenalg_networkit():
    for method in ["leidenalg", "networkit"]:
        for mode in ["smart", "naive", "raw"]:
            dynamic_launch("wiki_talk_ht", 10, method, mode = mode)

@pytest.mark.short  
def test_dynamic_magi():
    for mode in ["smart", "naive", "raw"]:
        dynamic_launch("wiki_talk_ht", 1, "magi", mode = mode)

@pytest.mark.long
def test_dynamic_dmon():
    for mode in ["smart", "naive", "raw"]:
        dynamic_launch("wiki_talk_ht", 1, "dmon", mode = mode)

"""
@pytest.mark.debug
def test_dynamic_networkit_bad_datasets():
    # Работает нестабильно, периодически 'зависает'
    for dataset in ["sociopatterns-hypertext", "radoslaw_email"]:
        dynamic_launch(dataset, 100, "networkit", mode = "smart", verbose = 2)
"""

def collect_diag(pid: int, out_dir: Path, duration_sec: float = 6.0):
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    err_path = out_dir / f"diag_errors_{pid}.log"
    errf = open(err_path, "a", buffering=1)

    def _spawn(cmd, outfile):
        try:
            return subprocess.Popen(
                cmd,
                stdout=open(outfile, "wb"),
                stderr=subprocess.STDOUT
            )
        except Exception as e:
            errf.write(f"spawn failed: {' '.join(cmd)} :: {e}\n")
            return None

    # strace
    if shutil.which("strace"):
        t = _spawn(["strace", "-f", "-p", str(pid)], out_dir / f"strace_{pid}.log")
        if t: tasks.append(t)
    else:
        errf.write("strace not found\n")

    # gdb batch backtrace
    if shutil.which("gdb"):
        t = _spawn(
            ["gdb", "-q", "-batch", "-ex", "set pagination off", "-ex", "thread apply all bt", "-p", str(pid)],
            out_dir / f"gdb_bt_{pid}.log"
        )
        if t: tasks.append(t)
    else:
        errf.write("gdb not found\n")

    # py-spy top (если есть)
    if shutil.which("py-spy"):
        t = _spawn(
            ["py-spy", "top", "--pid", str(pid), "--duration", str(duration_sec)],
            out_dir / f"pyspy_top_{pid}.log"
        )
        if t: tasks.append(t)
    else:
        errf.write("py-spy not found\n")

    # Подождать время сбора
    time.sleep(duration_sec)

    # Остановить фоновые сборщики
    for t in tasks:
        try:
            t.terminate()
        except Exception as e:
            errf.write(f"terminate failed: {e}\n")

    errf.close()

def run_with_timeout_subproc(args, timeout_sec):
    # безопасное окружение
    base = {}
    for k, v in os.environ.items():
        k = str(k)
        if "=" in k:
            continue
        base[k] = "" if v is None else str(v)

    base["OMP_NUM_THREADS"] = "1"
    base["MKL_NUM_THREADS"] = "1"
    base["NUMEXPR_NUM_THREADS"] = "1"
    base["OMP_WAIT_POLICY"] = "PASSIVE"
    base["KMP_BLOCKTIME"] = "0"
    base["CUDA_VISIBLE_DEVICES"] = ""
    base["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    base.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    proc = subprocess.Popen(
        [sys.executable, "-u", "networkit_isolated.py", *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=base
    )
    try:
        out, err = proc.communicate(timeout=timeout_sec)
        return proc.returncode, out, err
    except subprocess.TimeoutExpired:
        # 1) Python-стеки подпроцесса
        try:
            os.kill(proc.pid, signal.SIGUSR1)
            time.sleep(1.0)
        except ProcessLookupError:
            pass

        # 2) Автодиагностика (strace/gdb/py-spy) до terminate
        try:
            collect_diag(proc.pid, Path(".profiles"), duration_sec=6.0)
        except Exception as e:
            # если что-то пошло не так — просто продолжаем
            Path(".profiles").mkdir(exist_ok=True)
            (Path(".profiles") / f"diag_collect_failed_{proc.pid}.log").write_text(str(e))

        # 3) Завершение подпроцесса
        proc.terminate()
        try:
            out, err = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
        return -9, out, err


ERROR_PATTERNS = re.compile(r"(Traceback|ERROR|CRITICAL|Segmentation fault|AssertionError|CUDA error)", re.IGNORECASE)

@pytest.mark.debug
@pytest.mark.parametrize("repeat", range(15))
def test_dynamic_networkit_bad_datasets(resource_monitor, prof, repeat):
    profiles_dir = Path(".profiles")
    profiles_dir.mkdir(exist_ok=True)

    for dataset in ["sociopatterns-hypertext", "radoslaw_email"]:
        rc, out, err = run_with_timeout_subproc(
            [dataset, "100", "networkit", "smart", "2"], timeout_sec=30
        )
        if out:
            print(out)
        if err:
            print(err)

        if rc != 0:
            (profiles_dir / f"stack_repeat{repeat}_{dataset}.log").write_text(err or "", encoding="utf-8")
            if prof.is_running:
                prof.stop()
            prof.write_html(profiles_dir / f"profile_repeat{repeat}_{dataset}_fail.html")
            pytest.fail(f"Repeat {repeat} dataset {dataset} failed: rc={rc}")

        