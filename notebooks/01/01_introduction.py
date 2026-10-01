# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "scikit-learn",
# ]
# ///
#
# Lecture 1 (Oct 19), Session 1 — Introduction to machine learning.
# Run locally with `marimo edit notebooks/01/01_introduction.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).
#
# NOTE on scoping: Marimo requires each global name to be owned by exactly
# one cell. All cell-locals are underscore-prefixed. UI elements are created
# in one cell (whose *output* is the intro text — the md must be the last
# expression) and *read* in the following cell, which displays the widget
# together with its figure via mo.vstack. Figures are rendered with
# mo.image(BytesIO) — mo.as_html does not work in the Pyodide/WASM build.

import marimo

__generated_with = "0.17.6"
app = marimo.App(
    width="medium",
    layout_file="layouts/01_introduction.slides.json",
)


@app.cell
def _():
    import os
    from pathlib import Path

    if "__file__" in globals() and __file__:
        try:
            os.chdir(Path(__file__).resolve().parent)
        except OSError:
            pass

    import marimo as mo
    return mo, os, Path


@app.cell
def _(mo):
    mo.md(
        r"""
        # Introduction to Machine Learning

        **Machine Learning with Python** — Lecture 1, Session 1 (Oct 19)

        EUGLOH — *Problem Solving Using Open-Source Languages; R and Python*

        University of Novi Sad

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">1 / 17</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Today's session

        - **What is machine learning?** — the paradigm shift from rules to data
        - **Types of machine learning** — supervised, unsupervised, semi- and
          self-supervised, reinforcement
        - **The ML workflow** — and where the real work happens
        - **Over- and underfitting** — the central tension of the field

        Session 1 of 4 today. Please ask questions at any point.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">2 / 17</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Practical information

        - Slides are Marimo notebooks, exercises are Jupyter notebooks — all in this repository
        - Clone the repo and follow the setup in the `README.md` (`uv sync`, then `uv run jupyter lab`)
        - Today's exercises: `notebooks/01/01_introduction_{beginner,intermediate,advanced}.ipynb`
        - All exercises use `numpy`, `matplotlib` and `scikit-learn` — nothing else to install

        Questions are very welcome — ask early, ask often.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 17</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # What is machine learning?

        > *"A computer program is said to learn from experience E with respect to
        > some class of tasks T and performance measure P, if its performance at
        > tasks in T, as measured by P, improves with experience E."*
        — Mitchell, 1997

        In one sentence:

        **Machine learning is the discipline of building systems that learn rules
        from data, instead of being explicitly programmed.**

        - Traditional programming: *data + program → output*
        - Machine learning: *data + output → program* — the learned **model** is
          then applied to new, unseen data.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 17</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Why machine learning?

        - Problems that are **hard to specify** but easy to demonstrate:
          spam detection, image recognition, machine translation
        - Problems that **adapt over time**: recommender systems, fraud detection
        - Problems at **scale**, where hand-written rules break down

        ML is not magic — it is a tool that shines when the data is right and
        the question is well-posed.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 17</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Types of machine learning

        | Type | What the algorithm sees | Goal |
        |---|---|---|
        | **Supervised** | inputs **+ labels** | predict the label of new inputs |
        | **Unsupervised** | inputs only | discover structure (clusters, low-dimensional representations) |
        | **Semi-supervised** | few labels + many unlabeled inputs | leverage both |
        | **Self-supervised** | inputs only | create labels from the data itself (denoising, masked words) |
        | **Reinforcement** | interactions + rewards | learn actions that maximise reward |

        This course is mostly about **supervised learning**.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">6 / 17</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Supervised learning

        Given a dataset of (input, target) pairs $(x_i, y_i)$, learn a function
        $f(x) \approx y$.

        - **Classification** — $y$ is a discrete label (spam / not spam, healthy / diseased)
        - **Regression** — $y$ is a continuous number (price, temperature)

        Algorithms in this course: **linear & logistic regression** (session 2),
        **decision trees** (session 3), **random forests** (session 4), and
        **neural networks** (tomorrow).

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">7 / 17</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Unsupervised, semi-supervised, self-supervised

        - **Unsupervised:** only inputs. *Clustering* — group similar points;
          *dimensionality reduction* — find a compact representation.
        - **Semi-supervised:** a few labeled + many unlabeled points (labeling is
          often expensive — think medical images). The unlabeled points reveal
          the *shape* of the data, which constrains the decision boundary.
        - **Self-supervised:** the data creates its own supervision — predict a
          masked word, reconstruct a denoised image. The engine behind modern LLMs.
        - **Reinforcement learning:** an agent acts, the environment rewards.
          Out of scope for this course.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">8 / 17</div>
        """
    )
    return


@app.cell
def _(mo):
    # The widget is created here; the *next* cell reads `.value` and displays
    # the slider together with its figure. The md must be the LAST expression
    # so it becomes the cell's output.
    label_fraction = mo.ui.slider(
        start=0.0, stop=1.0, step=0.05, value=0.1,
        label="Fraction of labeled points", show_value=True, debounce=True,
    )
    mo.md(
        r"""
        ## One dataset, three paradigms

        The slider controls **how many points are labeled**:

        - 100 % labeled → **supervised** classification
        - a few labeled → **semi-supervised** learning
        - none labeled → **unsupervised** learning (e.g. clustering)

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">9 / 17</div>
        """
    )
    return (label_fraction,)


@app.cell
def _(label_fraction, mo):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.datasets import make_moons as _make_moons

    _X, _y = _make_moons(n_samples=150, noise=0.25, random_state=0)
    _n = len(_y)
    _k = int(round(label_fraction.value * _n))
    _order = _np.random.default_rng(0).permutation(_n)
    _labeled = _order[:_k]

    _fig, _ax = _plt.subplots(figsize=(6.5, 4.2))
    _ax.scatter(_X[:, 0], _X[:, 1], color="#cbd5e1", s=30, label="unlabeled")
    if _k > 0:
        _ax.scatter(
            _X[_labeled, 0], _X[_labeled, 1], c=_y[_labeled],
            cmap="RdYlGn", vmin=-0.15, vmax=1.15, s=80,
            edgecolor="k", linewidth=0.7, label="labeled",
        )
    if _k == 0:
        _mode = "Unsupervised — no labels: look for structure"
    elif _k < 0.5 * _n:
        _mode = f"Semi-supervised — only {_k}/{_n} points labeled"
    else:
        _mode = "Supervised — every point has a label"
    _ax.set_title(_mode)
    _ax.set_xticks([])
    _ax.set_yticks([])
    _ax.legend(loc="upper right")
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            label_fraction,
            mo.image(_buf, width="620px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">10 / 17</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The ML workflow

        1. **Frame the problem** — what is the task, what counts as success?
        2. **Collect & explore the data** — plots, summary statistics
        3. **Preprocess** — missing values, encoding, scaling
        4. **Split** — training / validation / test sets
        5. **Choose & train a model** — fit to the training set
        6. **Evaluate** — on held-out data, never on the training set
        7. **Iterate**

        Most of the real work happens in steps 2–3.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">11 / 17</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Overfitting and underfitting

        - **Underfitting:** the model is too simple — it misses the pattern (high *training* error).
        - **Overfitting:** the model memorises noise — great on training data, poor on new data.

        The cure: more data, simpler models, **regularisation** — and always
        keeping a **held-out test set** to detect it.

        Below: logistic regression on two moons with polynomial features of
        increasing degree — watch the boundary wiggle.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">12 / 17</div>
        """
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.datasets import make_moons as _make_moons
    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.pipeline import make_pipeline as _mkpipe
    from sklearn.preprocessing import PolynomialFeatures as _Poly
    from sklearn.preprocessing import StandardScaler as _Scaler

    _X, _y = _make_moons(n_samples=200, noise=0.3, random_state=1)
    _xx, _yy = _np.meshgrid(
        _np.linspace(_X[:, 0].min() - 0.5, _X[:, 0].max() + 0.5, 250),
        _np.linspace(_X[:, 1].min() - 0.5, _X[:, 1].max() + 0.5, 250),
    )
    _fig, _axes = _plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for _ax, _deg in zip(_axes, [1, 3, 9]):
        _pipe = _mkpipe(_Poly(degree=_deg), _Scaler(), _LR(max_iter=2000))
        _pipe.fit(_X, _y)
        _zz = _pipe.predict(_np.c_[_xx.ravel(), _yy.ravel()]).reshape(_xx.shape)
        _ax.contourf(_xx, _yy, _zz, alpha=0.25, cmap="RdYlGn")
        _ax.scatter(_X[_y == 0, 0], _X[_y == 0, 1], color="#dc2626", s=14)
        _ax.scatter(_X[_y == 1, 0], _X[_y == 1, 1], color="#16a34a", s=14)
        _ax.set_title(f"degree {_deg} · train acc {_pipe.score(_X, _y):.2f}")
        _ax.set_aspect("equal")
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.image(_buf, width="860px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">13 / 17</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The datasets of this course

        Everything runs offline — the datasets ship with scikit-learn.

        - **Synthetic** — `make_blobs`, `make_moons`, `make_circles` for clean illustrations
        - **Iris** — 150 flowers, 4 features, 3 species
        - **Wine** — 178 wines, 13 chemical features, 3 cultivars
        - **Digits** — 1797 handwritten 8×8 images, 10 classes
        - **Diabetes** — 442 patients, 10 features, a continuous target (regression)

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">14 / 17</div>
        """
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.datasets import (
        load_digits as _load_digits,
        load_iris as _load_iris,
        load_wine as _load_wine,
        make_blobs as _make_blobs,
        make_circles as _make_circles,
        make_moons as _make_moons,
    )

    _fig, _axes = _plt.subplots(2, 3, figsize=(11, 6.2))

    _Xb, _yb = _make_blobs(n_samples=150, centers=[(-2, -2), (2, 2)], cluster_std=1.2, random_state=0)
    _axes[0, 0].scatter(_Xb[:, 0], _Xb[:, 1], c=_yb, cmap="RdYlGn", s=14)
    _axes[0, 0].set_title("Gaussian blobs (synthetic)")

    _Xm, _ym = _make_moons(n_samples=150, noise=0.2, random_state=0)
    _axes[0, 1].scatter(_Xm[:, 0], _Xm[:, 1], c=_ym, cmap="RdYlGn", s=14)
    _axes[0, 1].set_title("Two moons (synthetic)")

    _Xc, _yc = _make_circles(n_samples=300, factor=0.4, noise=0.1, random_state=0)
    _axes[0, 2].scatter(_Xc[:, 0], _Xc[:, 1], c=_yc, cmap="RdYlGn", s=14)
    _axes[0, 2].set_title("Concentric circles (synthetic)")

    _iris = _load_iris()
    _axes[1, 0].scatter(_iris.data[:, 2], _iris.data[:, 3], c=_iris.target, cmap="RdYlGn", s=16)
    _axes[1, 0].set_title("Iris — petals (3 species)")

    _wine = _load_wine()
    _axes[1, 1].scatter(_wine.data[:, 0], _wine.data[:, 6], c=_wine.target, cmap="RdYlGn", s=16)
    _axes[1, 1].set_title("Wine — alcohol vs flavanoids")

    _digits = _load_digits()
    _axes[1, 2].imshow(_digits.images[0], cmap="gray_r")
    _axes[1, 2].set_title("Digits — 8×8 pixel images")

    for _ax in _axes.ravel():
        _ax.set_xticks([])
        _ax.set_yticks([])
    _fig.tight_layout()
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.image(_buf, width="860px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">15 / 17</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Summary

        - **Machine learning** learns rules from data instead of being programmed.
        - **Supervised** learning maps inputs to labels (classification) or
          numbers (regression); other paradigms (unsupervised, semi- and
          self-supervised, reinforcement) differ in what the algorithm sees.
        - The **ML workflow** is mostly data work; the held-out **test set** is
          sacred.
        - **Overfitting** (memorising) and **underfitting** (missing the pattern)
          are the two failure modes every model balances.

        Next session: **linear and logistic regression**.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">16 / 17</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Thanks for this session!

        Questions? Next up: **linear and logistic regression**.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">17 / 17</div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
