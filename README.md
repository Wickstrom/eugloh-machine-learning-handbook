# Machine Learning with Python

This repository contains the teaching material for the machine-learning module of the
[EUGLOH *Problem Solving Using Open-Source Languages; R and Python* course](https://www.eugloh.eu/courses-trainings/activities/problem-solving-using-open-source-languages-r-and-python/),
hosted by the **University of Novi Sad** and taught as a blended (online + on-site) course
within the [European University Alliance for Global Health (EUGLOH)](https://www.eugloh.eu/).

The material follows the same structure as the
[Pattern Recognition Handbook](https://github.com/Wickstrom/pattern-recognition-handbook)
by Kristoffer Wickstrøm, but is intentionally kept small and focused for a short,
introductory course.

- **Slides** are either Jupyter notebooks (rendered to static Reveal.js decks with
  `jupyter nbconvert --to slides`) or Marimo notebooks (exported to a self-contained
  WebAssembly app with `marimo export html-wasm`). Both are hosted on GitHub Pages.
- **Exercises** are classic Jupyter notebooks that you open and work in locally —
  clone the repo, fill in the cells, and run them.

## 📑 Content

1. **Introduction to machine learning**
    - What is machine learning?
    - Supervised, unsupervised, and reinforcement learning
    - The ML workflow: data, model, evaluation
    - Practical information for the course

    [![Slides](https://img.shields.io/badge/-Slides-blue?logo=jupyter&style=flat&labelColor=gray)](https://wickstrom.github.io/eugloh-machine-learning-handbook/notebooks/00/intro.slides.html)

2. **Linear and logistic regression**
    - Linear regression — the model, MSE loss, closed-form vs gradient descent
    - Logistic regression — sigmoid, cross-entropy, decision boundaries
    - Practical considerations: feature scaling, regularisation, multi-class
    - Interactive decision-boundary demo with a class-separation slider

    [![Slides](https://img.shields.io/badge/-Slides-blue?logo=marimo&style=flat&labelColor=gray)](https://wickstrom.github.io/eugloh-machine-learning-handbook/notebooks/01/linear_logistic_regression/index.html)

    **Exercises**

    - Synthetic regression in 2D
      [![Exercise](https://img.shields.io/badge/-Exercise-green?logo=jupyter&style=flat&labelColor=gray)](notebooks/01/regression_synthetic_exercise.ipynb)
    - Regression on the California housing dataset
      [![Exercise](https://img.shields.io/badge/-Exercise-green?logo=jupyter&style=flat&labelColor=gray)](notebooks/01/regression_california_exercise.ipynb)
    - Synthetic classification in 2D
      [![Exercise](https://img.shields.io/badge/-Exercise-green?logo=jupyter&style=flat&labelColor=gray)](notebooks/01/classification_synthetic_exercise.ipynb)
    - Classification on the Iris dataset
      [![Exercise](https://img.shields.io/badge/-Exercise-green?logo=jupyter&style=flat&labelColor=gray)](notebooks/01/classification_iris_exercise.ipynb)

## 💻 How to run the notebooks locally

We use [**uv**](https://docs.astral.sh/uv/) for fast, reproducible Python environments.

1. Install **uv** (one-time):

    ```bash
    # macOS / Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Windows (PowerShell)
    irm https://astral.sh/uv/install.ps1 | iex
    ```

2. Clone this repository and move into it:

    ```bash
    git clone https://github.com/Wickstrom/eugloh-machine-learning-handbook.git
    cd eugloh-machine-learning-handbook
    ```

3. Sync the dependencies (creates a `.venv` and installs everything in `pyproject.toml`):

    ```bash
    uv sync
    ```

4. Launch Jupyter Lab:

    ```bash
    uv run jupyter lab
    ```

5. To edit a Marimo notebook locally with live code, file watchers, and a variable explorer:

    ```bash
    uv run marimo edit notebooks/01/linear_logistic_regression.py
    ```

You only ever need `uv run` — it will use the `.venv` automatically, no need to
manually activate or deactivate anything.

## 🎞 Lecture slides (GitHub Pages)

Lecture material is published in two formats:

- **Jupyter notebooks** (`.ipynb`) — rendered to static Reveal.js slide decks with
  `jupyter nbconvert --to slides`. Same pipeline as [RISE](https://rise.readthedocs.io/en/latest/),
  nested horizontal/vertical slides, but static so they never time out.
- **Marimo notebooks** (`.py`) — exported to a self-contained WASM app with
  `marimo export html-wasm --mode run`. Runs entirely in the browser, including
  the Python interpreter; reactive widgets (sliders, dropdowns, tabs) update
  figures live.

**Live slides:** https://wickstrom.github.io/eugloh-machine-learning-handbook/

### Build locally

From the repo root, render everything with the same commands the CI uses:

```bash
# Jupyter notebooks -> Reveal.js slide decks
for nb in notebooks/*/*.ipynb; do
  base=$(basename "$nb" .ipynb)
  case "$base" in
    *_exercise) echo "Skipping $nb — exercise notebook"; continue ;;
  esac
  uv run jupyter nbconvert --to slides "$nb" --output-dir "_slides/$(dirname "$nb")"
done

# Marimo notebooks -> WASM apps
for nb in notebooks/*/*.py; do
  dir=$(dirname "$nb")
  base=$(basename "$nb" .py)
  target="_slides/$dir/$base"
  mkdir -p "$target"
  uv run marimo export html-wasm "$nb" -o "$target" --mode run --no-show-code
done
```

### Keyboard shortcuts (inside a deck)

| Key            | Action                       |
| -------------- | ---------------------------- |
| `←` / `→`      | previous / next slide        |
| `↑` / `↓`      | previous / next sub-slide    |
| `Space`        | advance                      |
| `f`            | fullscreen                   |
| `?`            | help overlay                 |

### Deployment

A GitHub Actions workflow at
[`.github/workflows/publish-slides.yml`](.github/workflows/publish-slides.yml)
rebuilds the slides on every push to `main` and publishes the output to GitHub
Pages via the official Pages API (`actions/deploy-pages`). One-time setup: in
**Settings → Pages → Source**, choose **"GitHub Actions"**.

## 🎥 Notebook format and RISE (local interactive slides)

For local lectures where you want to execute code live during the talk, the
notebooks can be presented interactively with [RISE](https://rise.readthedocs.io/en/latest/).
Open a notebook, select `View → Cell Toolbar → Slideshow`, and press the
`Enter/Exit RISE Slideshow` button at the top to enter presentation mode.

## 📝 Citation

If you are using this material in your courses or in your research, please
consider citing it as follows:

```bibtex
@misc{euglohmlhandbook2026,
  author       = {EUGLOH Machine Learning Handbook},
  title        = {Machine Learning with Python — EUGLOH Course Material},
  year         = {2026},
  howpublished = {Online},
  url          = {https://github.com/Wickstrom/eugloh-machine-learning-handbook}
}
```

This repository is heavily based on the
[Pattern Recognition Handbook](https://github.com/Wickstrom/pattern-recognition-handbook)
by Kristoffer Wickstrøm at UiT The Arctic University of Norway, which in turn
is based on [Time Series Analysis with Python](https://github.com/FilippoMB/python-time-series-handbook)
by Filippo Maria Bianchi.

## 📄 License

[MIT](./LICENSE)
