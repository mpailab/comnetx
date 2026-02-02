#!/usr/bin/env python3
import os
import json
from pathlib import Path
import numpy as np
from scipy import sparse
import joblib

JOBLIB_EXTS = {".joblib", ".pkl"} 

ROOT = Path(__file__).resolve().parents[1]
SMALL = ROOT / "test" / "graphs" / "small"
OUT_DIR = ROOT / "datasets-info"
OUT_PATH = OUT_DIR / "magi.json"

OVERRIDES = {
    # "acm": {"d": "undirected", "w": "unweighted"},
}

EDGE_EXTS = {".edgelist", ".edges", ".txt", ".csv"}
NP_EXTS = {".npy", ".npz"}

def infer_w_from_data(data):
    if data.size == 0:
        return "unweighted"
    has_neg = np.any(data < 0)
    all_ones = np.allclose(data, 1.0)
    if all_ones:
        return "unweighted"
    return "sign-weighted" if has_neg else "weighted"

def load_adj_from_npy(path: Path):
    arr = np.load(path, allow_pickle=True)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        try:
            mat = sparse.load_npz(path)
            return mat.tocsr()
        except Exception:
            pass
    if isinstance(arr, dict) and {"data","indices","indptr","shape"} <= set(arr.keys()):
        mat = sparse.csr_matrix((arr["data"], arr["indices"], arr["indptr"]), shape=tuple(arr["shape"]))
        return mat.tocsr()
    if sparse.issparse(arr):
        return arr.tocsr()
    if isinstance(arr, np.ndarray):
        return sparse.csr_matrix(arr)
    if hasattr(arr, "item"):
        try:
            obj = arr.item()
            if isinstance(obj, dict) and "adj" in obj:
                A = obj["adj"]
                if sparse.issparse(A):
                    return A.tocsr()
                return sparse.csr_matrix(A)
        except Exception:
            pass
    raise ValueError(f"Unsupported adjacency format in {path}")

def load_adj_from_joblib(path: Path):
    obj = joblib.load(path)

    # 0) Уже разреженная
    if sparse.issparse(obj):
        return obj.tocsr()

    if isinstance(obj, dict):
        # Ваш формат из PyTorch: {"indices": 2xNNZ, "values": NNZ, "shape": (n, m)}
        if {"indices", "values", "shape"} <= set(obj.keys()):
            idx = np.asarray(obj["indices"])
            # ожидаем форму (2, nnz); если (nnz, 2) — транспонируем
            if idx.ndim == 2 and idx.shape[0] == 2:
                row = idx[0]
                col = idx[1]
            elif idx.ndim == 2 and idx.shape[1] == 2:
                row = idx[:, 0]
                col = idx[:, 1]
            else:
                raise ValueError(f"indices has invalid shape: {idx.shape}")
            data = np.asarray(obj["values"])
            shape = tuple(obj["shape"])
            A = sparse.coo_matrix((data, (row, col)), shape=shape)
            return A.tocsr()
            
    # 1) CSR-компоненты
    if isinstance(obj, dict) and {"data","indices","indptr","shape"} <= set(obj.keys()):
        return sparse.csr_matrix((obj["data"], obj["indices"], obj["indptr"]),
                                 shape=tuple(obj["shape"])).tocsr()

    # 2) COO-компоненты как словарь
    if isinstance(obj, dict):
        # допускаем разные имена ключей
        keys = set(k.lower() for k in obj.keys())
        if {"row","col","data"} <= keys or {"i","j","data"} <= keys:
            get = lambda k: obj.get(k) or obj.get(k.upper()) or obj.get(k.capitalize())
            row = np.asarray(get("row") or get("i"))
            col = np.asarray(get("col") or get("j"))
            data = np.asarray(get("data"))
            shape = obj.get("shape")
            if shape is None:
                shape = (int(row.max())+1, int(col.max())+1) if row.size and col.size else (0, 0)
            A = sparse.coo_matrix((data, (row, col)), shape=tuple(shape))
            return A.tocsr()
        # вложенный ключ 'adj'
        if "adj" in obj:
            A = obj["adj"]
            if sparse.issparse(A):
                return A.tocsr()
            if isinstance(A, np.ndarray):
                return sparse.csr_matrix(A)

    # 3) COO-компоненты как кортеж/список
    if isinstance(obj, (tuple, list)):
        # ожидаем (data, (row, col)) или (data, (row, col), shape)
        if len(obj) in (2, 3):
            data = np.asarray(obj[0])
            ij = obj[1]
            if isinstance(ij, (tuple, list)) and len(ij) == 2:
                row = np.asarray(ij[0]); col = np.asarray(ij[1])
                shape = tuple(obj[2]) if len(obj) == 3 else (int(row.max())+1, int(col.max())+1)
                A = sparse.coo_matrix((data, (row, col)), shape=shape)
                return A.tocsr()

    # 4) Плотная матрица/массив
    if isinstance(obj, np.ndarray):
        return sparse.csr_matrix(obj)

    raise ValueError(f"Unsupported joblib object in {path} (type={type(obj)})")


def find_adj_in_dir(ds_dir: Path):
    candidates = [p for p in ds_dir.iterdir()
                  if p.is_file() and (p.suffix.lower() in (NP_EXTS | JOBLIB_EXTS))]
    candidates.sort(key=lambda p: 0 if "adj" in p.name.lower() else 1)
    for p in candidates:
        try:
            if p.suffix.lower() in JOBLIB_EXTS:
                A = load_adj_from_joblib(p)
            else:
                A = load_adj_from_npy(p)
            if A.shape[0] == A.shape[1]:
                return A, p
        except Exception as e:
            print(f"[warn] {ds_dir.name}: {p.name} skipped: {e}")
    return None, None

def analyze_adj(A: sparse.csr_matrix):
    n = A.shape[0]
    A_no_diag = A.tocsr(copy=True)
    A_no_diag.setdiag(0)
    A_no_diag.eliminate_zeros()
    m = int(A_no_diag.nnz)
    is_symmetric = False
    if A_no_diag.shape[0] == A_no_diag.shape[1]:
        diff = (A_no_diag - A_no_diag.T).nnz
        is_symmetric = (diff == 0)
    d = "undirected" if is_symmetric else "directed"
    if d == "undirected":
        m = m // 2
    w = infer_w_from_data(A_no_diag.data if A_no_diag.nnz > 0 else np.array([]))
    return n, m, d, w

def apply_overrides(name, rec):
    o = OVERRIDES.get(name, {})
    if "d" in o:
        rec["d"] = o["d"]
    if "w" in o:
        rec["w"] = o["w"]
    return rec

def analyze_dataset_dir(ds_dir: Path):
    A, src = find_adj_in_dir(ds_dir)
    if A is not None:
        n, m, d, w = analyze_adj(A)
        return {"n": n, "m": m, "d": d, "w": w}

    from csv import DictReader, reader

    def infer_sign_weighted(weights):
        if not weights:
            return "unweighted"
        arr = np.array([w for w in weights if w is not None], dtype=float)
        if arr.size == 0:
            return "unweighted"
        if np.any(arr < 0):
            return "sign-weighted"
        if np.allclose(arr, 1.0):
            return "unweighted"
        return "weighted"

    nodes = set()
    m = 0
    weights = []
    directed_guess = False

    for fn in ds_dir.iterdir():
        if not fn.is_file():
            continue
        ext = fn.suffix.lower()
        if ext not in EDGE_EXTS:
            continue

        if ext == ".csv":
            with open(fn, "r", encoding="utf-8", newline="") as f:
                dr = DictReader(f)
                cols = [c.lower() for c in (dr.fieldnames or [])]
                if {"source", "target"}.issubset(cols):
                    idx = {c.lower(): c for c in dr.fieldnames}
                    for r in dr:
                        u = r[idx["source"]]
                        v = r[idx["target"]]
                        w = r.get(idx.get("weight", ""), "")
                        w = float(w) if w not in ("", None) else None
                        nodes.add(u); nodes.add(v); m += 1; weights.append(w)
                    directed_guess = True
                    continue

            with open(fn, "r", encoding="utf-8", newline="") as f:
                rd = reader(f)
                _ = next(rd, None)
                for row in rd:
                    if len(row) < 2:
                        continue
                    u, v = row[0], row[1]
                    w = float(row[2]) if len(row) > 2 and row[2] != "" else None
                    nodes.add(u); nodes.add(v); m += 1; weights.append(w)
            directed_guess = True

        else:
            with open(fn, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    parts = s.split()
                    if len(parts) < 2:
                        continue
                    u, v = parts[0], parts[1]
                    w = None
                    if len(parts) >= 3:
                        try:
                            w = float(parts[2])
                        except:
                            w = None
                    nodes.add(u); nodes.add(v); m += 1; weights.append(w)
            directed_guess = False

    if m > 0:
        return {
            "n": len(nodes),
            "m": m,
            "d": "directed" if directed_guess else "undirected",
            "w": infer_sign_weighted(weights),
        }

    print(f"[skip] {ds_dir.name}: adjacency not found and no edgelist/csv")
    return None

def build_index():
    if not SMALL.exists():
        raise FileNotFoundError(f"Not found: {SMALL}")
    result = {}
    for ds_dir in sorted([p for p in SMALL.iterdir() if p.is_dir()]):
        name = ds_dir.name
        rec = analyze_dataset_dir(ds_dir)
        if rec is None:
            continue
        rec = apply_overrides(name, rec)
        result[name] = rec
    return result

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = build_index()
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=4)
    print(f"Wrote: {OUT_PATH}")

if __name__ == "__main__":
    main()
