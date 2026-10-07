# Machine Learning with Python

This repository contains the teaching material for the machine-learning module of the
[EUGLOH *Problem Solving Using Open-Source Languages; R and Python* course](https://www.eugloh.eu/courses-trainings/activities/problem-solving-using-open-source-languages-r-and-python/),
hosted by the **University of Novi Sad** and taught as a blended (online + on-site) course
within the [European University Alliance for Global Health (EUGLOH)](https://www.eugloh.eu/).

- **Slides** are [Marimo](https://marimo.io/) notebooks (`.py`), exported to a
  self-contained WebAssembly app with `marimo export html-wasm` and hosted on
  GitHub Pages. Reactive widgets (sliders, dropdowns, tabs) update figures live
  in the browser — no Python kernel required.
- **Exercises** are classic Jupyter notebooks (`.ipynb`) in three difficulty
  levels, one set per session. Open them locally (see
  [How to run the notebooks locally](#-how-to-run-the-notebooks-locally)) or run
  them in the cloud with **Binder** — no setup required. See the
  [Exercises](#-exercises) section for details.

## 📑 Content

Each day is split into **four sessions** (one Marimo deck per session), following
the course schedule.

### Day 1 — Oct 19

1. **Session 1 — Introduction to machine learning** (09:00)
    - What is ML? Supervised, unsupervised, semi- and self-supervised learning
    - The ML workflow, over- and underfitting, train/validation/test splits
    - The datasets used throughout the course
    - Interactive demo: labeling paradigms (supervised → semi-supervised → unsupervised)

    [![Slides](https://img.shields.io/badge/-Slides-blue?logo=marimo&style=flat&labelColor=gray)](https://wickstrom.github.io/eugloh-machine-learning-handbook/notebooks/01/01_introduction/index.html)

2. **Session 2 — Linear & logistic regression** (10:00)
    - Linear regression — MSE, the closed-form fit, and gradient descent
    - The 1801 story of **Piazzi, Ceres and Gauss** — least squares as one of
      the first uses of learning from data
    - Logistic regression — sigmoid, cross-entropy, gradient descent,
      implemented **by hand in numpy** and compared against scikit-learn
    - Multi-class with one-vs-rest / softmax
    - Interactive demos: watching the fit update as **new samples arrive**
      (linear and logistic), scrubbing through gradient descent, and tuning
      the **regularisation strength $C$**

    [![Slides](https://img.shields.io/badge/-Slides-blue?logo=marimo&style=flat&labelColor=gray)](https://wickstrom.github.io/eugloh-machine-learning-handbook/notebooks/01/02_regression/index.html)

3. **Session 3 — Decision trees** (11:15)
    - Impurity (Gini/entropy), CART, one split computed **by hand**
    - Interpretability — reading a tree as rules, and where it breaks down
    - Overfitting, pruning, complexity
    - Interactive demos: choosing a split (weighted impurity vs threshold),
      tree depth with and without label noise

    [![Slides](https://img.shields.io/badge/-Slides-blue?logo=marimo&style=flat&labelColor=gray)](https://wickstrom.github.io/eugloh-machine-learning-handbook/notebooks/01/03_decision_trees/index.html)

4. **Session 4 — Random forests & ensembles** (12:15)
    - Bagging, boosting, feature importance
    - Why trees dominate tabular machine learning
    - Interactive demo: number of trees and depth

    [![Slides](https://img.shields.io/badge/-Slides-blue?logo=marimo&style=flat&labelColor=gray)](https://wickstrom.github.io/eugloh-machine-learning-handbook/notebooks/01/04_random_forests/index.html)

### Day 2 — Oct 20

5. **Session 1 — Introduction to neural networks** (09:00)
    - From linear to non-linear classifiers; the **artificial neuron** and
      **activation functions**; the **XOR problem**; a brief history

    [![Slides](https://img.shields.io/badge/-Slides-blue?logo=marimo&style=flat&labelColor=gray)](https://wickstrom.github.io/eugloh-machine-learning-handbook/notebooks/02/01_introduction/index.html)

6. **Session 2 — Multilayer networks** (10:00)
    - The big idea, what a **hidden layer** does to the data, the **forward
      pass**, and the **MLP**
    - Interactive demo: move an input through the network and watch it become separable

    [![Slides](https://img.shields.io/badge/-Slides-blue?logo=marimo&style=flat&labelColor=gray)](https://wickstrom.github.io/eugloh-machine-learning-handbook/notebooks/02/02_perceptron_mlp/index.html)

7. **Session 3 — Forward/backward pass & optimization** (11:15)
    - Forward pass, backpropagation, gradient descent and the learning rate
    - An MLP **written from scratch in numpy**, compared against `MLPClassifier`

    [![Slides](https://img.shields.io/badge/-Slides-blue?logo=marimo&style=flat&labelColor=gray)](https://wickstrom.github.io/eugloh-machine-learning-handbook/notebooks/02/03_training/index.html)

8. **Session 4 — Components & going beyond** (12:15)
    - Components: weight initialization, optimizers (SGD/momentum/Adam),
      regularization, capacity
    - Training in practice: reading the curves, **feature scaling**, and the
      common **failure modes**
    - Going beyond: softmax, convolutional networks, **transformers**, autoencoders
    - Interactive demo: hidden-layer-size and activation-function widgets

    [![Slides](https://img.shields.io/badge/-Slides-blue?logo=marimo&style=flat&labelColor=gray)](https://wickstrom.github.io/eugloh-machine-learning-handbook/notebooks/02/04_components_and_beyond/index.html)

## 📚 Exercises

Every session comes with a set of **Jupyter notebooks** in three difficulty
levels — pick the one that matches your comfort with Python. All three levels
work on the **same data** and build toward the same idea, so you can start at
one level and move up.

**Run them all in the cloud (no local setup):**

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Wickstrom/eugloh-machine-learning-handbook/main?urlpath=lab/tree/notebooks)

This opens JupyterLab in [Binder](https://mybinder.org/) with every exercise
notebook under `notebooks/`. The **first** launch is slow (Binder builds and
caches the environment, usually a few minutes); later launches reuse the cached
image and start in seconds. The environment is defined by [`binder/`](binder/)
(`requirements.txt` + `runtime.txt`) and kept small so the build stays quick.

**The three levels**

| Level | What you do | What you practise |
|---|---|---|
| **Beginner** | Explore and understand the data — load it, summarise it, plot it | `pandas`, `matplotlib` |
| **Intermediate** | Use the library — fit a model with scikit-learn and evaluate it | the `scikit-learn` API, accuracy, confusion matrix |
| **Advanced** | Build the core algorithm yourself, from scratch | the maths behind the model, in `numpy` |

**How they are structured**

- One notebook per session per level, named
  `notebooks/<day>/<session>_<level>.ipynb` — for example
  `notebooks/02/03_training_advanced.ipynb`, `notebooks/01/02_regression_intermediate.ipynb`.
- Every notebook has the **same five tasks**, so the levels line up: each task
  has a short explanation, a cell with a `# TODO` for you to fill in, and a
  **Wrap-up** at the end to check your understanding.
- Work through them in order. The beginner notebook is the gentlest entry point;
  the advanced one is where the algorithm finally becomes yours.

## 💻 How to run the notebooks locally

The recommended path: clone the repo, sync the environment with `uv`, and
launch Jupyter Lab locally.

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
    uv run marimo edit notebooks/01/01_introduction.py
    ```

You only ever need `uv run` — it will use the `.venv` automatically, no need to
manually activate or deactivate anything.

To run the **exercises** instead, open the `.ipynb` files in JupyterLab — or use
the [Binder link](#-exercises) above to skip the local setup entirely.

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
uv run marimo edit notebooks/01/01_introduction.py
uv run marimo edit notebooks/02/01_introduction.py
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

This repository is heavily based on:

- [Pattern Recognition Handbook](https://github.com/Wickstrom/pattern-recognition-handbook) by Kristoffer Wickstrøm:

    ```bibtex
    @misc{wick2025prbook,
      author       = {Kristoffer Wickstr{\o}m},
      title        = {Pattern Recognition Course},
      year         = {2025},
      howpublished = {Online},
      url          = {https://github.com/Wickstrom/pattern-recognition-handbook}
    }
    ```

- [Time Series Analysis with Python](https://github.com/FilippoMB/python-time-series-handbook) by Filippo Maria Bianchi:

    ```bibtex
    @misc{bianchi2024tsbook,
      author       = {Filippo Maria Bianchi},
      title        = {Time Series Analysis with Python},
      year         = {2024},
      howpublished = {Online},
      url          = {https://github.com/FilippoMB/python-time-series-handbook}
    }
    ```
