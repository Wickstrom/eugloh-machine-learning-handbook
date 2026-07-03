# AGENTS.md

EUGLOH machine-learning course material. Marimo lectures (`.py`) + Jupyter exercises (`.ipynb`), with a one-job CI that exports Marimo decks to WASM and publishes them on GitHub Pages.

## Setup — uv only, no conda

```
uv sync
uv run jupyter lab                                          # exercise notebooks
uv run marimo edit notebooks/00/intro.py                    # edit a lecture locally
uv run marimo edit notebooks/01/linear_logistic_regression.py
```

Never run `pip install` / `conda env create` / `source .venv/bin/activate` — always `uv run` so the venv is picked up automatically. If `uv` is missing: `curl -LsSf https://astral.sh/uv/install.sh | sh`.

## Repo layout

```
notebooks/
  00/intro.py                            # Marimo lecture (WASM)
  00/layouts/intro.slides.json           # required marimo slide layout
  01/linear_logistic_regression.py       # Marimo lecture (WASM)
  01/layouts/linear_logistic_regression.slides.json
  01/{regression,classification}_{synthetic,california,iris}_exercise.ipynb
index.html                               # copied to _slides/index.html by CI
.github/workflows/publish-slides.yml     # single CI job, builds + deploys
pyproject.toml                           # [tool.uv] package = false; Python >=3.10
```

`_slides/` is build output and gitignored — never commit it.

## Conventions the CI cares about

1. **Slides are Marimo only.** No `.ipynb` Reveal/RISE-style decks — every lecture lives at `notebooks/NN/<name>.py` with a matching layout file.
2. **Exercise notebooks must end in `_exercise.ipynb`** — the workflow leaves them alone. They are run locally with `uv run jupyter lab`, not built into the site.
3. **Each Marimo notebook needs a layout file** at `notebooks/NN/layouts/<base>.slides.json` with body `{"type": "slides", "data": {}}`. The CI inlines it as base64 into the exported HTML, so the JSON itself is not shipped. Missing → `marimo export html-wasm` errors.

The first cell of every Marimo notebook does `os.chdir(Path(__file__).resolve().parent)` so `media/` paths resolve the same in `marimo edit` and the deployed WASM build. Keep it.

## Build slides locally (same commands as CI)

```bash
mkdir -p _slides
for nb in notebooks/*/*.py; do
  dir=$(dirname "$nb"); base=$(basename "$nb" .py)
  target="_slides/$dir/$base"
  mkdir -p "$target"
  uv run marimo export html-wasm "$nb" -o "$target" --mode run --no-show-code
  rm -f "$target/CLAUDE.md"   # marimo >=0.23 ships one — strip it
done
cp index.html _slides/index.html && touch _slides/.nojekyll
```

Output: `_slides/notebooks/NN/<base>/index.html` + `assets/` per lecture, plus `_slides/index.html` as the landing page.

## Verification — there is no test suite

No pytest, ruff, mypy, or pre-commit. To check changes:

- **Marimo lectures**: `uv run marimo edit <file>.py` for interactive smoke-test, or `uv run marimo export html-wasm <file>.py -o /tmp/x` to validate the export pipeline. The marimo notebook must parse as Python and export without errors — `marimo export` is the de-facto typecheck.
- **Exercise notebooks**: cells with `# YOUR CODE HERE` raise `TypeError: cannot unpack non-iterable ellipsis object` if executed unfilled. Fill them in (or write a standalone script that replicates the solution) to confirm the expected outputs in the comments match.
- **Hand-authored `.ipynb`**: every cell needs an `id` field (nbformat 5.1+). After writing, fill any missing ids:
  ```python
  import nbformat, uuid
  nb = nbformat.read(p, as_version=4)
  for c in nb.cells:
      if not c.get("id"):
          c["id"] = uuid.uuid4().hex[:8]
  nbformat.write(nb, p)
  ```
  (`nbformat.normalize` does not exist in nbformat 5.x — generate ids manually as shown.)
- **Exercise notebook "Open in cloud" badges**: the second cell of every `*_exercise.ipynb` must be a markdown cell with id `open-in-cloud`. **Colab is the primary default** (official `colab-badge.svg`, points at the GitHub blob URL); **Binder is the smaller secondary option** (introduced by the text "No Google account?", points at `?urlpath=lab/tree/...`). Both services ship sklearn/numpy/pandas/matplotlib; no `%pip install` cell is needed. Badge URLs bake in `Wickstrom/eugloh-machine-learning-handbook` and `main` — if the repo moves, regenerate the badges.

## Gotchas

- **Marimo >=0.23 dumps `CLAUDE.md` into every export** (from its `_static/`). The workflow strips it; if you export locally, do the same.
- **GitHub URLs in `README.md` are wired to the `Wickstrom/eugloh-machine-learning-handbook` repo.** If you fork or rename, search-and-replace in `README.md`: `wickstrom.github.io/eugloh-machine-learning-handbook` (Pages URL) and `github.com/Wickstrom/eugloh-machine-learning-handbook` (clone URL + citation).
- **Marimo's `__generated_with = "0.17.6"`** in the lecture file is informational only — it does not need to match the installed marimo version.
- **PEP 723 script header** at the top of each `.py` Marimo notebook declares its own deps (`marimo`, `numpy`, …). Keep it in sync with what the notebook imports — `uv run marimo edit` warns on drift but does not fail.
- **`fetch_california_housing` and `load_iris`** come bundled with scikit-learn — no network needed at runtime.

## Deployment

One workflow (`.github/workflows/publish-slides.yml`) builds `_slides/` on every push to `main` or `master` (and on `workflow_dispatch`) and deploys via `actions/deploy-pages`. One-time repo setting: **Settings → Pages → Source → GitHub Actions**.
