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
# Lecture 2 (Oct 20), Session 2 — Multilayer networks.
# The big idea (stacking neurons), what a hidden layer does to the data, the
# forward pass, and the multi-layer perceptron. The artificial neuron, activation
# functions and the XOR problem are introduced in Session 1.
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
        # Multilayer Networks

        **Machine Learning with Python** — Lecture 2, Session 2 (Oct 20)

        EUGLOH — *Problem Solving Using Open-Source Languages; R and Python*

        University of Novi Sad

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">1 / 9</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Today's session

        - **The big idea** — stacking neurons to learn a useful transformation
        - **What a hidden layer does** — reshaping the data until it is separable
        - **The forward pass** — computing activations layer by layer
        - **The multi-layer perceptron (MLP)** — layers, shapes, matrix products

        Session 2 of 4 today. (The neuron, activation functions and the XOR
        problem were covered in Session 1.)

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">2 / 9</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Part 1 — Multilayer networks

        ## The big idea

        - A single neuron is a **linear classifier** — it needs linearly
          separable data and cannot solve **XOR** (Session 1).
        - What if we could **transform** the data into a representation where
          it *becomes* linearly separable?
        - And who computes that transformation? **Another neuron!**
        - Stack layers of neurons → a **multi-layer perceptron (MLP)**.

        The layers learn the transformation; the last layer separates.
        Everything is trained end-to-end with one algorithm: **backpropagation**.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 9</div>
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

        $$\text{hidden} = \phi(W_1 x + b_1), \qquad
        \hat{p} = \sigma(W_2 \cdot \text{hidden} + b_2)$$

        Every layer simply feeds its output to the next, so the whole network
        is a chain of weighted sums and activations.

        That chain is **matrix multiplication** — the reason GPUs are so good at
        this. (One hidden layer with enough units can approximate *any*
        continuous function — but nobody tells you how many units; you learn
        that from the data.)

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 9</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The forward pass, step by step

        Feed an input $x$ through the network:

        $$z^{(1)} = W_1 x + b_1, \quad a^{(1)} = \phi(z^{(1)}), \quad
        z^{(2)} = W_2 a^{(1)} + b_2, \quad \hat{p} = \sigma(z^{(2)})$$

        - Each line is a **weighted sum (+ bias)** followed by the **activation** — exactly the single neuron from Session 1.
        - A whole batch is one matrix product: stack the inputs into $X$ and compute $Z^{(1)} = X W_1 + b_1$.
        - In plain words: **signals flow forward**, layer by layer, from inputs to prediction.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 9</div>
        """
    )
    return


@app.cell
def _(mo):
    x1_slider = mo.ui.slider(
        start=-2.0, stop=2.0, step=0.05, value=0.9,
        label="input $x_1$", show_value=True, debounce=True,
    )
    x2_slider = mo.ui.slider(
        start=-2.0, stop=2.0, step=0.05, value=-0.9,
        label="input $x_2$", show_value=True, debounce=True,
    )
    mo.md(
        r"""
        ## What a hidden layer does — see it live

        The same **XOR** data, shown in two spaces. On the left, the original
        input space: the network's boundary has to bend, so no single line works.
        On the right, the *same* points after one hidden layer ($\phi = \tanh$):
        they have been **pulled apart into a linearly separable arrangement**.

        Drag the sliders to move a point through the network and watch its
        hidden activation — that is the **forward pass**.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">6 / 9</div>
        """
    )
    return x1_slider, x2_slider


@app.cell
def _(mo, x1_slider, x2_slider):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.neural_network import MLPClassifier as _MLP
    from sklearn.preprocessing import StandardScaler as _Scaler

    _rng = _np.random.default_rng(0)
    _centers = _np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    _labels = _np.array([0, 1, 1, 0])
    _n, _std = 60, 0.13
    _X = _np.vstack([_c + _std * _rng.standard_normal((_n, 2)) for _c in _centers])
    _y = _np.concatenate([_np.full(_n, _l) for _l in _labels])
    _sc = _Scaler().fit(_X)
    _Xs = _sc.transform(_X)

    _mlp = _MLP(
        hidden_layer_sizes=(2,), activation="tanh",
        solver="lbfgs", max_iter=5000, random_state=0,
    ).fit(_Xs, _y)
    _A1 = _np.tanh(_Xs @ _mlp.coefs_[0] + _mlp.intercepts_[0])
    _lr = _LR().fit(_A1, _y)

    # forward pass for the point chosen by the sliders
    _q = _np.array([[x1_slider.value, x2_slider.value]])
    _qa = _np.tanh(_q @ _mlp.coefs_[0] + _mlp.intercepts_[0])
    _prob = _mlp.predict_proba(_q)[0, 1]

    _xx, _yy = _np.meshgrid(
        _np.linspace(_Xs[:, 0].min() - 0.4, _Xs[:, 0].max() + 0.4, 220),
        _np.linspace(_Xs[:, 1].min() - 0.4, _Xs[:, 1].max() + 0.4, 220),
    )
    _Zin = _mlp.predict(_np.c_[_xx.ravel(), _yy.ravel()]).reshape(_xx.shape)

    _hx, _hy = _np.meshgrid(
        _np.linspace(_A1[:, 0].min() - 0.3, _A1[:, 0].max() + 0.3, 220),
        _np.linspace(_A1[:, 1].min() - 0.3, _A1[:, 1].max() + 0.3, 220),
    )
    _Zh = _lr.predict(_np.c_[_hx.ravel(), _hy.ravel()]).reshape(_hx.shape)

    _fig, _axes = _plt.subplots(1, 2, figsize=(11, 4.4))
    _axes[0].contourf(_xx, _yy, _Zin, alpha=0.22, cmap="RdYlGn")
    _axes[0].scatter(_Xs[_y == 0, 0], _Xs[_y == 0, 1], color="#dc2626", s=16)
    _axes[0].scatter(_Xs[_y == 1, 0], _Xs[_y == 1, 1], color="#16a34a", s=16)
    _axes[0].scatter([_q[0, 0]], [_q[0, 1]], marker="*", s=260,
                     color="#2563eb", edgecolor="k", zorder=5)
    _axes[0].set_title("Input space — curved boundary")
    _axes[0].set_aspect("equal")

    _axes[1].contourf(_hx, _hy, _Zh, alpha=0.22, cmap="RdYlGn")
    _axes[1].scatter(_A1[_y == 0, 0], _A1[_y == 0, 1], color="#dc2626", s=16)
    _axes[1].scatter(_A1[_y == 1, 0], _A1[_y == 1, 1], color="#16a34a", s=16)
    _axes[1].scatter([_qa[0, 0]], [_qa[0, 1]], marker="*", s=260,
                     color="#2563eb", edgecolor="k", zorder=5)
    _axes[1].set_title("After one hidden layer — linearly separable")
    _axes[1].set_xlabel("$a_1$")
    _axes[1].set_ylabel("$a_2$")
    _axes[1].set_aspect("equal")

    _fig.suptitle(
        f"forward pass:  hidden activation = ({_qa[0, 0]:.2f}, {_qa[0, 1]:.2f})"
        f"   →   output $\\hat p$ = {_prob:.2f}"
    )
    _fig.tight_layout()
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.hstack([x1_slider, x2_slider]),
            mo.image(_buf, width="900px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">7 / 9</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Summary

        - A single linear unit cannot solve **XOR** — but a **hidden layer** can
          transform the inputs so the last layer separates them.
        - The **forward pass** sends the signal through the network:
          weighted sums and activations, layer by layer.
        - An **MLP** stacks neurons in layers; the whole computation is a
          sequence of **matrix products**, trained end-to-end by
          **backpropagation**.

        Next session: the **forward pass, backward pass and optimization** — how
        the network actually *learns* its weights.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">8 / 9</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Where to go next

        - **Exercise:** `notebooks/02/02_perceptron_mlp_{beginner,intermediate,advanced}.ipynb`
          — build an MLP, inspect its hidden-layer representation, and watch the
          decision boundary bend.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">9 / 9</div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
