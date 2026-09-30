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
# Lecture 2 (Oct 20), Session 1 — Introduction to neural networks.
# Run locally with `marimo edit notebooks/02/01_introduction.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).
#
# NOTE on scoping: Marimo requires each global name to be owned by exactly
# one cell. All cell-locals are underscore-prefixed. UI elements are created
# in one cell (which shows only the intro text) and *read* in the following
# cell, which displays the widget together with its figure via mo.vstack —
# this keeps every interactive demo on a single slide.

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
        # Introduction to Neural Networks

        **Machine Learning with Python** — Lecture 2, Session 1 (Oct 20)

        EUGLOH — *Problem Solving Using Open-Source Languages; R and Python*

        University of Novi Sad

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">1 / 7</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Today's session

        - **From linear to non-linear** classifiers — why we need them
        - A brief **history** of neural networks — and the two AI winters
        - What we will build over the next three sessions: the **perceptron**,
          **multi-layer networks**, **backpropagation**, and where the field
          goes next (**CNNs**, **transformers**)

        Session 1 of 4 today.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">2 / 7</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Part 1 — The basics

        ## Introduction

        - So far our classifiers have drawn **straight lines** (logistic
          regression) or **axis-aligned cuts** (trees).
        - Many real problems need **smooth, curved boundaries**.

        Our focus now shifts to **non-linear classifiers** — and we start
        with the most influential one: the neural network.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 7</div>
        """
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.datasets import make_circles as _make_circles
    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.preprocessing import StandardScaler as _Scaler

    _X, _y = _make_circles(n_samples=400, factor=0.4, noise=0.1, random_state=0)
    nn_X, nn_y = _X, _y
    nn_Xs = _Scaler().fit_transform(_X)  # scaled copy — used for training

    _clf = _LR().fit(nn_Xs, nn_y)

    _xx, _yy = _np.meshgrid(
        _np.linspace(nn_Xs[:, 0].min() - 0.5, nn_Xs[:, 0].max() + 0.5, 250),
        _np.linspace(nn_Xs[:, 1].min() - 0.5, nn_Xs[:, 1].max() + 0.5, 250),
    )
    _Z = _clf.predict(_np.c_[_xx.ravel(), _yy.ravel()]).reshape(_xx.shape)

    _fig, _ax = _plt.subplots(figsize=(5.4, 4.6))
    _ax.contourf(_xx, _yy, _Z, alpha=0.25, cmap="RdYlGn")
    _ax.scatter(nn_Xs[nn_y == 0, 0], nn_Xs[nn_y == 0, 1], color="#dc2626", s=16, label="class 0")
    _ax.scatter(nn_Xs[nn_y == 1, 0], nn_Xs[nn_y == 1, 1], color="#16a34a", s=16, label="class 1")
    _ax.set_title(f"Concentric circles: logistic regression gets stuck at a line (acc {_clf.score(nn_Xs, nn_y):.2f})")
    _ax.set_aspect("equal")
    _ax.legend()
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.image(_buf, width="620px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 7</div>"""),
        ]
    )
    return nn_X, nn_Xs, nn_y


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Neural networks — a brief history

        - **McCulloch–Pitts neuron** (1943) — the first mathematical model of a neuron
        - **The perceptron** (Rosenblatt, 1957) — a learning algorithm!
        - **The XOR problem** (Minsky & Papert, 1969) — the first AI winter
        - **Backpropagation** (Rumelhart, Hinton & Williams, 1986) — a new hope
        - Lacked data and compute; hard to train (1990s) — the second AI winter
        - **AlexNet** (2012) — GPUs + ImageNet → the deep-learning revolution
        - **Transformers / LLMs** (2017–) — the same ideas, at scale

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 7</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Summary

        - Linear models (regression) and axis-aligned trees cannot capture
          **smooth, curved** boundaries — we need non-linear classifiers.
        - Neural networks are the most influential family of non-linear models;
          their history runs from the **McCulloch–Pitts neuron** through two
          **AI winters** to today's **transformers**.
        - Over the next three sessions we build up the **perceptron**, the
          **MLP**, **backpropagation**, and the practical **components** that
          make networks train.

        Next session: the **perceptron and multilayer networks**.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">6 / 7</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Thanks for this session!

        Questions? Next up: the **perceptron and multilayer networks**.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">7 / 7</div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
