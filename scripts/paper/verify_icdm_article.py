#!/usr/bin/env python3
"""Verify the ICDM article PDF and source hygiene."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_MARKERS = [
    "TBD",
    "TODO",
    "FIXME",
    "XXX",
    "cn69",
    "missing log",
    "/workspace",
    "/Users",
    "github.com",
    "subcoms_depth",
    "neighborhood_step",
    "Full ARI",
    "final-NMI deltas",
    "150.7",
    "3.64",
    "Compared solutions",
    "consolidated bundle",
    "measurement pipeline",
    "Bokov",
    "Konovalov",
    "Moscow",
    "MSU",
    "Lomonosov",
    "Russia",
    "Russian",
]


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"required tool is not available: {name}")


def fail(message: str, failures: list[str]) -> None:
    print(f"FAIL: {message}")
    failures.append(message)


def pass_(message: str) -> None:
    print(f"PASS: {message}")


def build_pdf(repo: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    for script in (
        "scripts/paper/plot_workload_speedup.py",
        "scripts/paper/plot_topology_ablation_pareto.py",
    ):
        plot = run([sys.executable, script], cwd=repo)
        if plot.returncode != 0:
            raise RuntimeError(
                f"failed to regenerate figure with {script}\n"
                + plot.stdout
                + plot.stderr
            )

    article_dir = repo / "article"
    for index in (1, 2):
        proc = run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={out_dir}",
                "article.tex",
            ],
            cwd=article_dir,
        )
        (out_dir / f"build{index}.out").write_text(proc.stdout + proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(f"pdflatex pass {index} failed; see {out_dir}")

    return out_dir / "article.pdf"


def pdf_pages(pdf: Path) -> int:
    proc = run(["pdfinfo", str(pdf)])
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    match = re.search(r"^Pages:\s+(\d+)$", proc.stdout, re.M)
    if not match:
        raise RuntimeError("could not parse page count from pdfinfo")
    return int(match.group(1))


def pdf_info(pdf: Path) -> str:
    proc = run(["pdfinfo", str(pdf)])
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    return proc.stdout


def extract_pdf_text(pdf: Path, text_path: Path) -> str:
    proc = run(["pdftotext", str(pdf), str(text_path)])
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    return text_path.read_text(errors="ignore")


def scan_markers(label: str, text: str, markers: list[str], failures: list[str]) -> None:
    found = [marker for marker in markers if marker in text]
    if found:
        fail(f"{label} contains blocked markers: {', '.join(found)}", failures)
    else:
        pass_(f"{label} marker scan")


def verify_no_appendix(tex: str, failures: list[str]) -> None:
    appendix_patterns = [
        r"\\appendix\b",
        r"\\begin\{appendices\}",
        r"\\section\*?\{Appendix",
    ]
    hits = [pattern for pattern in appendix_patterns if re.search(pattern, tex)]
    if hits:
        fail(f"main article contains appendix commands: {', '.join(hits)}", failures)
    else:
        pass_("no appendix in main article source")


def verify_anonymous_author_blocks(tex: str, failures: list[str]) -> None:
    names = re.findall(r"\\IEEEauthorblockN\{([^}]*)\}", tex)
    affiliations = re.findall(r"\\IEEEauthorblockA\{([^}]*)\}", tex)
    non_anonymous = [
        value
        for value in names + affiliations
        if value.strip() and value.strip() != "Anonymous"
    ]
    if non_anonymous:
        fail(f"non-anonymous author block values: {', '.join(non_anonymous)}", failures)
    elif not names:
        fail("no IEEE author blocks found", failures)
    else:
        pass_(f"anonymous IEEE author blocks ({len(names)} authors)")


def verify_pdf_metadata(info: str, markers: list[str], failures: list[str]) -> None:
    scan_markers("PDF metadata", info, markers, failures)
    author_match = re.search(r"^Author:\s*(.*)$", info, re.M)
    if author_match and author_match.group(1).strip() not in {"", "Anonymous"}:
        fail(f"PDF Author metadata is not anonymous: {author_match.group(1).strip()}", failures)
    else:
        pass_("PDF Author metadata")


def verify_hardware_reporting(tex: str, pdf_text: str, failures: list[str]) -> None:
    hardware_patterns = [
        r"wall-clock measurements were collected on",
        r"wall-clock measurements were run on",
        r"experiments were conducted on",
        r"experiments were run on",
    ]
    setup_match = re.search(
        r"\\subsection\{Experimental setup\}([\s\S]*?)\\subsection\{Experimental results\}",
        tex,
    )
    setup_tex = setup_match.group(1) if setup_match else ""
    has_source_sentence = any(re.search(pattern, setup_tex, re.I) for pattern in hardware_patterns)
    has_pdf_sentence = any(re.search(pattern, pdf_text, re.I) for pattern in hardware_patterns)
    if not (has_source_sentence and has_pdf_sentence):
        fail("final mode requires a hardware configuration sentence in the article", failures)
        return

    no_gpu_phrase = "no NVIDIA GPU was visible"
    gpu_terms = ("NVIDIA", "CUDA", "GPU-enabled baselines")
    if no_gpu_phrase.lower() in setup_tex.lower() or no_gpu_phrase.lower() in pdf_text.lower():
        fail("final mode cannot use a no-GPU hardware sentence for GPU-baseline experiments", failures)
    elif not all(term in setup_tex for term in ("NVIDIA", "CUDA")):
        fail("final mode requires NVIDIA GPU and CUDA details in the hardware sentence", failures)
    elif not any(term in pdf_text for term in gpu_terms):
        fail("final mode requires GPU details to appear in the PDF text", failures)
    else:
        pass_("hardware configuration reported")


def verify_latex_log(log_path: Path, failures: list[str]) -> None:
    log = log_path.read_text(errors="ignore")
    checks = {
        "Overfull \\hbox": "overfull boxes",
        "LaTeX Warning": "LaTeX warnings",
        "undefined": "undefined references",
        "multiply defined": "multiply-defined references",
    }
    for needle, name in checks.items():
        count = log.count(needle)
        if count:
            fail(f"{name}: {count}", failures)
        else:
            pass_(name)


def verify_labels_and_cites(tex: str, min_references: int, failures: list[str]) -> None:
    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", tex))
    missing_refs = sorted(refs - labels)
    if missing_refs:
        fail(f"missing labels: {', '.join(missing_refs)}", failures)
    else:
        pass_(f"labels resolved ({len(refs)} references)")

    bibitems = set(re.findall(r"\\bibitem\{([^}]+)\}", tex))
    cites: set[str] = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", tex):
        cites.update(item.strip() for item in group.split(",") if item.strip())

    missing_bibitems = sorted(cites - bibitems)
    uncited = sorted(bibitems - cites)
    if missing_bibitems:
        fail(f"missing bibitems: {', '.join(missing_bibitems)}", failures)
    else:
        pass_(f"citations resolved ({len(cites)} citation keys)")
    if uncited:
        fail(f"uncited bibitems: {', '.join(uncited)}", failures)
    else:
        pass_("all bibliography entries are cited")
    if len(bibitems) < min_references:
        fail(f"only {len(bibitems)} references, expected at least {min_references}", failures)
    else:
        pass_(f"reference count >= {min_references} ({len(bibitems)})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/pdfs/icdm_verify"))
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--min-references", type=int, default=50)
    parser.add_argument(
        "--final",
        action="store_true",
        help="Enable final-submission checks that require measurement-server hardware reporting.",
    )
    parser.add_argument("--keep-tmp", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else repo / args.out_dir
    failures: list[str] = []

    for tool in ("pdflatex", "pdfinfo", "pdftotext"):
        require_tool(tool)

    pdf = build_pdf(repo, out_dir)
    info = pdf_info(pdf)
    pages = pdf_pages(pdf)
    if pages > args.max_pages:
        fail(f"page count {pages} exceeds limit {args.max_pages}", failures)
    else:
        pass_(f"page count {pages}/{args.max_pages}")

    verify_latex_log(out_dir / "article.log", failures)

    article_tex = (repo / "article" / "article.tex").read_text(errors="ignore")
    scan_markers("article source", article_tex, DEFAULT_MARKERS, failures)
    verify_no_appendix(article_tex, failures)
    verify_anonymous_author_blocks(article_tex, failures)
    verify_pdf_metadata(info, DEFAULT_MARKERS, failures)
    pdf_text = extract_pdf_text(pdf, out_dir / "article.txt")
    scan_markers("article PDF", pdf_text, DEFAULT_MARKERS, failures)
    if args.final:
        verify_hardware_reporting(article_tex, pdf_text, failures)
    verify_labels_and_cites(article_tex, args.min_references, failures)

    if not args.keep_tmp:
        shutil.rmtree(out_dir, ignore_errors=True)

    if failures:
        print(f"\n{len(failures)} verification check(s) failed.")
        return 1

    print("\nICDM article verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
