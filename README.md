# Machine Learning with Python

This repository contains the teaching material for the machine-learning module of the
[EUGLOH *Problem Solving Using Open-Source Languages; R and Python* course](https://www.eugloh.eu/courses-trainings/activities/problem-solving-using-open-source-languages-r-and-python/),
hosted by the **University of Novi Sad** and taught as a blended (online + on-site) course
within the [European University Alliance for Global Health (EUGLOH)](https://www.eugloh.eu/).

The material follows the same structure as the
[Pattern Recognition Handbook](https://github.com/Wickstrom/pattern-recognition-handbook)
by Kristoffer Wickstrøm, but is intentionally kept small and focused for a short,
introductory course.

- **Slides** are [Marimo](https://marimo.io/) notebooks (`.py`), exported to a
  self-contained WebAssembly app with `marimo export html-wasm` and hosted on
  GitHub Pages. Reactive widgets (sliders, dropdowns, tabs) update figures live
  in the browser — no Python kernel required.
- **Exercises** are classic Jupyter notebooks (`.ipynb`) that you open and work
  in locally — clone the repo, fill in the cells, and run them.

## 📑 Content

1. **Introduction to machine learning**
    - What is machine learning?
    - Supervised, unsupervised, and reinforcement learning
    - The ML workflow: data, model, evaluation
    - Practical information for the course

    [![Slides](https://img.shields.io/badge/-Slides-blue?logo=marimo&style=flat&labelColor=gray)](https://wickstrom.github.io/eugloh-machine-learning-handbook/notebooks/00/intro/index.html)

2. **Linear and logistic regression**
    - Linear regression — the model, MSE loss, closed-form vs gradient descent
    - Logistic regression — sigmoid, cross-entropy, decision boundaries
    - Practical considerations: feature scaling, regularisation, multi-class
    - Interactive decision-boundary demo with a class-separation slider

    [![Slides](https://img.shields.io/badge/-Slides-blue?logo=marimo&style=flat&labelColor=gray)](https://wickstrom.github.io/eugloh-machine-learning-handbook/notebooks/01/linear_logistic_regression/index.html)

    **Exercises** &mdash; every notebook has *Open in Colab* and *Binder* badges at the top, so students can start working without any local setup. Local setup (below) is recommended for an environment that matches `pyproject.toml` exactly.

    - Synthetic regression in 2D
      [![Exercise](https://img.shields.io/badge/-Exercise-green?logo=jupyter&style=flat&labelColor=gray)](notebooks/01/regression_synthetic_exercise.ipynb)
    - Regression on the California housing dataset
      [![Exercise](https://img.shields.io/badge/-Exercise-green?logo=jupyter&style=flat&labelColor=gray)](notebooks/01/regression_california_exercise.ipynb)
    - Synthetic classification in 2D
      [![Exercise](https://img.shields.io/badge/-Exercise-green?logo=jupyter&style=flat&labelColor=gray)](notebooks/01/classification_synthetic_exercise.ipynb)
    - Classification on the Iris dataset
      [![Exercise](https://img.shields.io/badge/-Exercise-green?logo=jupyter&style=flat&labelColor=gray)](notebooks/01/classification_iris_exercise.ipynb)

## 💻 How to run the notebooks locally

The recommended path: clone the repo, sync the environment with `uv`, and
launch Jupyter Lab locally.

> **No local setup?** Every exercise notebook also has *Open in Colab* and
> *Binder* badges at the top. Both services ship scikit-learn, numpy, pandas,
> and matplotlib pre-installed, so students can start immediately without
> installing anything. Colab is the fastest option (a few seconds); Binder takes
> 1&ndash;2 minutes the first time and uses the exact `pyproject.toml`.

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

Each lecture is a Marimo notebook that is exported with
`marimo export html-wasm --mode run` to a self-contained WebAssembly app.
Everything runs in the browser — Python interpreter, dependencies, code, and
all — so the slides never time out and need no kernel on the server. Give them
a few seconds to spin up on first load.

**Live slides:** https://wickstrom.github.io/eugloh-machine-learning-handbook/

### Build locally

From the repo root, render every Marimo notebook with the same commands the CI uses:

```bash
for nb in notebooks/*/*.py; do
  dir=$(dirname "$nb")
  base=$(basename "$nb" .py)
  target="_slides/$dir/$base"
  mkdir -p "$target"
  uv run marimo export html-wasm "$nb" -o "$target" --mode run --no-show-code
done
```

To edit a lecture locally with live code, file watchers, and a variable
explorer:

```bash
uv run marimo edit notebooks/00/intro.py
uv run marimo edit notebooks/01/linear_logistic_regression.py
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
