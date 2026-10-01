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
# Lecture 2 (Oct 20), Session 2 — The perceptron and multilayer networks.
# Run locally with `marimo edit notebooks/02/02_perceptron_mlp.py`
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
    layout_file="layouts/02_perceptron_mlp.slides.json",
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
        # The Perceptron & Multilayer Networks

        **Machine Learning with Python** — Lecture 2, Session 2 (Oct 20)

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

        - **The artificial neuron** — a weighted sum plus a non-linear activation
        - **The XOR problem** — the limitation of a single linear unit
        - **The big idea** — stacking neurons to learn a useful transformation
        - **The multi-layer perceptron (MLP)** — layers, shapes, matrix products

        Session 2 of 4 today.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">2 / 7</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Part 1 — The perceptron and multilayer networks

        ## The artificial neuron

        A neuron first forms a **weighted sum** of its inputs, then squeezes the
        result through a non-linear **activation**:

        $$z = w_1 x_1 + w_2 x_2 + \dots + b, \qquad a = \phi(z)$$

        In plain words: multiply each input by its **weight**, add them up
        (plus a **bias** $b$), then pass the total through a curve $\phi$.

        The **perceptron** is exactly this unit with a threshold/sigmoid
        activation. Recognise it? **Logistic regression is exactly one neuron**
        with the sigmoid activation (Lecture 1, Session 2). A neural network is
        *many neurons stacked in layers*, each feeding the next.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 7</div>
        """
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np

    _rng = _np.random.default_rng(42)
    _n, _std = 50, 0.15
    _centers = _np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    _labels = _np.array([0, 1, 1, 0])
    _X = _np.vstack([_c + _std * _rng.standard_normal((_n, 2)) for _c in _centers])
    _y = _np.concatenate([_np.full(_n, _l) for _l in _labels])

    _fig, _ax = _plt.subplots(figsize=(4.6, 4.6))
    _ax.scatter(_X[_y == 0, 0], _X[_y == 0, 1], color="#dc2626", s=60,
                edgecolor="k", alpha=0.75, label="class 0")
    _ax.scatter(_X[_y == 1, 0], _X[_y == 1, 1], color="#16a34a", s=60,
                edgecolor="k", alpha=0.75, label="class 1")
    _ax.set_aspect("equal")
    _ax.set_xlim(-0.55, 1.55)
    _ax.set_ylim(-0.55, 1.55)
    _ax.set_title("The XOR problem — no single line can solve it")
    _ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
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
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The big idea

        - A perceptron (a linear classifier) requires **linearly separable** data.
        - What if we could **transform** the data into a representation where
          it *becomes* linearly separable?
        - And who computes that transformation? **Another perceptron!**
        - Stack layers of perceptrons → a **multi-layer perceptron (MLP)**.

        The layers learn the transformation; the last layer separates.
        Everything is trained end-to-end with one algorithm: **backpropagation**.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 7</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The multi-layer perceptron (MLP)

        Layers of neurons: **input → hidden → output**. For one hidden layer,
        each hidden unit takes a weighted sum of the inputs and applies the
        activation:

        $$\text{hidden} = \tanh(W_1 x + b_1), \qquad
        \hat{p} = \sigma(W_2 \cdot \text{hidden} + b_2)$$

        Every layer simply feeds its output to the next, so the whole network
        is a chain of weighted sums and activations.

        That chain is **matrix multiplication** — the reason GPUs are so good at
        this. (One hidden layer with enough units can approximate *any*
        continuous function — but nobody tells you how many units; you learn
        that from the data.)

        The activation $\phi$ is a *component* we will study tomorrow — for now
        keep $\tanh$ in the hidden layer and $\sigma$ at the output.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">6 / 7</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Summary

        - A **neuron** is a weighted sum plus a non-linear activation;
          **logistic regression is one neuron**.
        - A single linear unit cannot solve **XOR** — but a **hidden layer** can
          transform the inputs so the last layer separates them.
        - An **MLP** stacks neurons in layers; the whole computation is a
          sequence of **matrix products**, trained end-to-end by
          **backpropagation**.

        Next session: the **forward pass, backward pass and optimization**.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">7 / 7</div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
