# AGENTS.md

EUGLOH machine-learning course material. Static-content repo with two notebook formats and a one-job CI that publishes slides to GitHub Pages.

## Setup — uv only, no conda

```
uv sync
uv run jupyter lab                                      # exercise notebooks
uv run marimo edit notebooks/01/linear_logistic_regression.py   # marimo slide deck
```

Never run `pip install` / `conda env create` / `source .venv/bin/activate` — always `uv run` so the venv is picked up automatically. If `uv` is missing: `curl -LsSf https://astral.sh/uv/install.sh | sh`.

## Repo layout

```
notebooks/
  00/intro.ipynb                              # Reveal.js slides (nbconvert)
  01/linear_logistic_regression.py            # Marimo slides (WASM)
  01/layouts/<base>.slides.json               # required marimo slide layout
  01/{regression,classification}_{synthetic,california,iris}_exercise.ipynb
index.html                                    # copied to _slides/index.html by CI
.github/workflows/publish-slides.yml          # single CI job, builds + deploys
pyproject.toml                                # [tool.uv] package = false; Python >=3.10
```

`_slides/` is build output and gitignored — never commit it.

## Three notebook conventions the CI cares about

1. **Exercise notebooks must end in `_exercise.ipynb`** — the workflow skips them. Anything else without a same-named `.py` is converted by nbconvert.
2. **A `.py` marimo notebook shadows its `.ipynb` sibling** — if both `intro.py` and `intro.ipynb` exist in the same folder, only the `.py` is exported (to a WASM directory at `_slides/notebooks/NN/<base>/index.html`).
3. **Each `.py` marimo notebook needs a layout file** at `notebooks/NN/layouts/<base>.slides.json` with body `{"type": "slides", "data": {}}`. The CI inlines it as base64 into the exported HTML, so the JSON itself is not shipped. Missing → `marimo export html-wasm` errors.

The first cell of every `.py` marimo notebook does `os.chdir(Path(__file__).resolve().parent)` so `media/` paths resolve the same in `marimo edit` and the deployed WASM build. Keep it.

## Build slides locally (same commands as CI)

```bash
mkdir -p _slides
for nb in notebooks/*/*.ipynb; do
  base=$(basename "$nb" .ipynb)
  case "$base" in *_exercise) continue ;; esac
  [ -f "$(dirname "$nb")/$base.py" ] && continue
  uv run jupyter nbconvert --to slides "$nb" \
    --output-dir "_slides/$(dirname "$nb")" --output "$base"
done

for nb in notebooks/*/*.py; do
  dir=$(dirname "$nb"); base=$(basename "$nb" .py)
  target="_slides/$dir/$base"
  mkdir -p "$target"
  uv run marimo export html-wasm "$nb" -o "$target" --mode run --no-show-code
  rm -f "$target/CLAUDE.md"   # marimo >=0.23 ships one — strip it
done

cp index.html _slides/index.html && touch _slides/.nojekyll
```

Output is `_slides/notebooks/NN/<base>.slides.html` (nbconvert) or `_slides/notebooks/NN/<base>/index.html` + `assets/` (marimo).

## Verification — there is no test suite

No pytest, ruff, mypy, or pre-commit. To check changes:

- **`.ipynb` slides**: `uv run jupyter nbconvert --to notebook --execute <file> --output /tmp/out.ipynb` to run end-to-end.
- **Exercise notebooks**: cells with `# YOUR CODE HERE` raise `TypeError: cannot unpack non-iterable ellipsis object` if executed unfilled. Fill them in (or write a standalone script that replicates the solution) to confirm the expected outputs in the comments match.
- **`.py` marimo**: `uv run marimo edit <file>.py` for interactive smoke-test, or `uv run marimo export html-wasm <file>.py -o /tmp/x` to validate the export pipeline.
- **Hand-authored `.ipynb`**: every cell needs an `id` field (nbformat 5.1+). After writing, `python -c "import nbformat; nb=nbformat.read(p, as_version=4); nbformat.normalize(nb); nbformat.write(nb, p)"`.
- **`.ipynb` slide metadata**: RISE uses `{"slideshow": {"slide_type": "slide" | "subslide" | "fragment" | "skip"}}` on each markdown cell. Without it the deck renders as a single wall.

## Gotchas

- **Marimo >=0.23 dumps `CLAUDE.md` into every export** (from its `_static/`). The workflow strips it; if you export locally, do the same.
- **GitHub URLs in `README.md` are wired to the `Wickstrom/eugloh-machine-learning-handbook` repo.** If you fork or rename, search-and-replace in `README.md`: `wickstrom.github.io/eugloh-machine-learning-handbook` (Pages URL) and `github.com/Wickstrom/eugloh-machine-learning-handbook` (clone URL + citation).
- **Marimo's `__generated_with = "0.17.6"`** in the lecture file is informational only — it does not need to match the installed marimo version.
- **PEP 723 script header** at the top of each `.py` Marimo notebook declares its own deps (`marimo`, `numpy`, …). Keep it in sync with what the notebook imports — `uv run marimo edit` warns on drift but does not fail.
- **`fetch_california_housing` and `load_iris`** come bundled with scikit-learn — no network needed at runtime.

## Deployment

One workflow (`.github/workflows/publish-slides.yml`) builds `_slides/` on every push to `main` or `master` (and on `workflow_dispatch`) and deploys via `actions/deploy-pages`. One-time repo setting: **Settings → Pages → Source → GitHub Actions**.
