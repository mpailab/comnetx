import os
import subprocess

def get_visible_physical_gpus():
    """
    Возвращает список физических GPU (index, uuid, name),
    которые соответствуют текущему CUDA_VISIBLE_DEVICES.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cvd is None:
        mapping = None
    else:
        mapping = [int(x) for x in cvd.split(",")]

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name",
            "--format=csv,noheader"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )

    lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
    all_gpus = []
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        idx = int(parts[0])
        uuid = parts[1]
        name = ",".join(parts[2:])
        all_gpus.append((idx, uuid, name))

    if mapping is None:
        return all_gpus
    else:
        return [all_gpus[i] for i in mapping]

def print_logical_to_physical():
    from textwrap import indent

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    print(f"CUDA_VISIBLE_DEVICES = {cvd}")

    visible_physical = get_visible_physical_gpus()

    print("Logical -> physical mapping:")
    for logical_id, (phys_idx, uuid, name) in enumerate(visible_physical):
        print(f"  cuda:{logical_id} -> GPU {phys_idx} ({uuid}, {name})")

if __name__ == "__main__":
    print_logical_to_physical()
