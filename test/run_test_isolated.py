"""
Run all pytest test files in the current directory,
capture results, and print a summary.
"""

import subprocess
import sys
import os
import re
from collections import defaultdict

current_dir = os.path.dirname(os.path.abspath(__file__))

pytest_args = sys.argv[1:]

test_files = [
    f for f in os.listdir(current_dir)
    if f.startswith("test_") and f.endswith(".py")
]

test_results = defaultdict(list)

test_line_re = re.compile(r'(.+\.py)::(\S+)\s+(PASSED|FAILED|SKIPPED|ERROR)')

for test_file in test_files:
    env = os.environ.copy()
    env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "pytest", test_file, "--tb=short", "--disable-warnings", "-v"] + pytest_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env
    )

    for line in proc.stdout:
        print(line, end="")

        match = test_line_re.match(line.strip())
        if match:
            file_name, test_name, status = match.groups()
            test_results[file_name].append((test_name, status))

    proc.wait()
    if proc.returncode != 0:
        pass

total = passed = failed = skipped = errors = 0
print("\n\n=== TEST SUMMARY ===")
for file_name, tests in test_results.items():
    print(f"{file_name}:")
    for test_name, status in tests:
        print(f"  - {test_name}: {status}")
        total += 1
        if status == "PASSED":
            passed += 1
        elif status == "FAILED":
            failed += 1
        elif status == "SKIPPED":
            skipped += 1
        elif status == "ERROR":
            errors += 1

print("\n=== TOTALS ===")
print(f"Total: {total}, Passed: {passed}, Failed: {failed}, Skipped: {skipped}, Errors: {errors}")

if failed > 0 or errors > 0:
    raise RuntimeError(f"{failed + errors} test(s) failed or errored.")
