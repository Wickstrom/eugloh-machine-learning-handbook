#!/usr/bin/env python3
"""Build slide PDFs for every Marimo lecture deck into ``pdfs/``.

Usage (from the repo root)::

    uv run python scripts/build_pdfs.py            # every deck
    uv run python scripts/build_pdfs.py notebooks/02/02_perceptron_mlp.py

Requirements::

    uv pip install "nbconvert[webpdf]" playwright
    playwright install chromium

Notes
-----
* We use marimo's default ``--raster-server=static``. The docs recommend
  ``--raster-server=live`` for slides, but with these widget/two-column decks the
  live capture drops slide content (symptom: fully blank pages) — static is both
  faster and correct here.
* Code inputs are excluded (``--no-include-inputs``) to match the published
  WebAssembly slides, which are built with ``--no-show-code``.
* After each export the script reports page count and any suspicious pages
  (near-white, or leaking a machine path from a captured stderr warning).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# A page counts as blank when it is almost entirely white.
WHITE_MEAN = 250.0
WHITE_STD = 3.0


def _cleanup(pdf: Path) -> None:
    """Report page count and drop artifact pages (best effort; needs poppler).

    Two kinds of page are removed, both of which are *extra* pages beyond the
    deck's slide count (verified against the decks' ``N / M`` footers):

    * fully blank pages, and
    * pages that are nothing but a leaked stderr block (e.g. a
      ``ConvergenceWarning`` captured when the export ran the notebook).
    """
    import numpy as np

    try:
        import pikepdf
        from PIL import Image
    except ImportError:
        return

    with tempfile.TemporaryDirectory() as tmp:
        try:
            subprocess.run(
                ["pdftoppm", "-png", "-r", "36", str(pdf), str(Path(tmp) / "p")],
                check=True,
                capture_output=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return

        pages = sorted(Path(tmp).glob("p*.png"))
        blank, warn = [], []
        for i, p in enumerate(pages, 1):
            arr = np.asarray(Image.open(p).convert("L"), dtype=float)
            if arr.mean() >= WHITE_MEAN and arr.std() <= WHITE_STD:
                blank.append(i)
            text = subprocess.run(
                ["pdftotext", "-f", str(i), "-l", str(i), "-layout", str(pdf), "-"],
                capture_output=True,
                text=True,
            ).stdout
            if "/site-packages/" in text or "ConvergenceWarning" in text:
                warn.append(i)

        drop = sorted(set(blank) | set(warn))
        if drop:
            with pikepdf.open(pdf) as doc:
                keep = [i for i in range(len(doc.pages)) if (i + 1) not in drop]
                out = pikepdf.Pdf.new()
                for i in keep:
                    out.pages.append(doc.pages[i])
                out.save(pdf)

        flags = []
        if blank:
            flags.append(f"blank {blank}")
        if warn:
            flags.append(f"stderr {warn}")
        tail = f"  !! dropped {'; '.join(flags)}" if drop else "  ok"
        print(f"  {len(pages)} -> {len(pages) - len(drop)} pages{tail}")


def build(nb: Path) -> Path:
    nb = nb.resolve()
    out = ROOT / "pdfs" / nb.parent.name / f"{nb.stem}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"== {nb.relative_to(ROOT)} -> {out.relative_to(ROOT)} ==")
    subprocess.run(
        [
            sys.executable, "-m", "marimo", "export", "pdf", str(nb),
            "-o", str(out), "--as=slides", "--no-include-inputs", "-f",
        ],
        cwd=nb.parent,
        check=True,
    )
    _cleanup(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("notebooks", nargs="*", type=Path, help="decks to build (default: all)")
    args = ap.parse_args()

    for nb in args.notebooks or sorted(ROOT.glob("notebooks/*/*.py")):
        build(nb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
