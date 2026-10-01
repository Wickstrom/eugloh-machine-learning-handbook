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
# Lecture 2 (Oct 20), Session 3 — Forward pass, backward pass and optimization.
# An MLP written from scratch in numpy; sklearn for comparison; gradient descent
# and the learning rate.
# Run locally with `marimo edit notebooks/02/03_training.py`
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
    layout_file="layouts/03_training.slides.json",
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
        # Forward Pass, Backward Pass & Optimization

        **Machine Learning with Python** — Lecture 2, Session 3 (Oct 20)

        EUGLOH — *Problem Solving Using Open-Source Languages; R and Python*

        University of Novi Sad

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">1 / 14</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Today's session

        - **The forward pass** — computing activations layer by layer
        - **The backward pass** — backpropagation: the chain rule, layer by layer
        - **Optimization** — gradient descent and the learning rate
        - An MLP **written from scratch in numpy** — and compared with sklearn

        Session 3 of 4 today.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">2 / 14</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Part 1 — Training: forward, backward, update

        ## Training: loss + backpropagation

        Same recipe as logistic regression — cross-entropy loss, gradient
        descent. The new skill is computing the gradient *through the
        layers*: the **chain rule**, applied layer by layer from the output
        back to the input — **backpropagation**:

        $$\delta_2 = \frac{\hat{p} - y}{n} \quad \text{(output error)}$$

        $$\frac{\partial \mathcal{L}}{\partial W_2} = A_1^\top \delta_2, \qquad \frac{\partial \mathcal{L}}{\partial b_2} = \textstyle\sum_i \delta_{2,i}$$

        $$\delta_1 = (\delta_2 W_2^\top) \odot (1 - A_1^2) \quad \text{(through the tanh derivative)}$$

        $$\frac{\partial \mathcal{L}}{\partial W_1} = X^\top \delta_1, \qquad \frac{\partial \mathcal{L}}{\partial b_1} = \textstyle\sum_i \delta_{1,i}$$

        You will implement exactly these lines — by hand — below and in the exercise.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 14</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The backpropagation recipe

        1. **Initialise** the parameters (small random weights)
        2. **Forward pass** — compute all activations
        3. **Backward pass** — compute the gradients with the chain rule
        4. **Update** all parameters with gradient descent
        5. **Repeat!**

        Top tips for implementing it:

        - **Keep it simple in the beginning** — start with a fixed architecture
        - **Do the calculations by hand first**, cross-reference while coding
        - There is only **one loop** — the epochs; everything inside is
          **matrix multiplication**

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 14</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Optimization — how we actually descend

        Backpropagation gives us the **gradients**; **optimization** decides how
        to use them. The simplest rule is **gradient descent**:

        $$w \leftarrow w - \eta \, \nabla_w \mathcal{L}$$

        - $\eta$ is the **learning rate** — *the* single most important
          hyperparameter.
          - too small → training crawls
          - too large → the loss oscillates or **diverges**
        - One pass over the whole training set is an **epoch**.
        - **Batch / mini-batch / stochastic** gradient descent trade the
          accuracy of each step against its cost — mini-batches are the
          workhorse of deep learning.

        More sophisticated **optimizers** (momentum, Adam, …) build on this
        idea; we will meet them in the next session.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 14</div>
        """
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np

    _lrs = [0.05, 0.2, 0.6, 0.95, 1.02]
    _colors = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#f59e0b"]

    _fig, _axes = _plt.subplots(1, 2, figsize=(12, 4))
    _wgrid = _np.linspace(-5, 5, 200)
    _axes[0].plot(_wgrid, _wgrid**2, color="gray", lw=2)
    _axes[0].set_xlabel("$w$")
    _axes[0].set_ylabel(r"$\mathcal{L}(w) = w^2$")
    _axes[0].set_title("A simple convex loss to descend")

    for _lr, _c in zip(_lrs, _colors):
        _w = 4.0
        _hist = [_w]
        for _ in range(25):
            _w = _w - _lr * 2 * _w
            _hist.append(_w)
        _hist = _np.clip(_hist, -12, 12)
        _axes[1].plot(_hist, "o-", ms=3, color=_c, label=f"lr = {_lr}")

    _axes[1].axhline(0, color="gray", ls="--", lw=0.8)
    _axes[1].set_xlabel("iteration")
    _axes[1].set_ylabel("$w$")
    _axes[1].set_title("Too large a learning rate over/under-shoots")
    _axes[1].legend(fontsize=8)
    _axes[1].grid(alpha=0.3)
    _fig.tight_layout()
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.image(_buf, width="880px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">6 / 14</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    import numpy as _np
    from sklearn.datasets import make_circles as _make_circles
    from sklearn.preprocessing import StandardScaler as _Scaler

    _X, _y = _make_circles(n_samples=400, factor=0.4, noise=0.1, random_state=0)
    nn_Xs = _Scaler().fit_transform(_X)
    nn_y = _y

    mo.md(
        r"""
        ## An MLP by hand, in numpy

        2 inputs → 16 hidden units (tanh) → 1 output (sigmoid), trained with
        full-batch gradient descent on the circles dataset (standardised).

        TODO: walk through the forward pass, then the gradient lines.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">7 / 14</div>
        """
    )
    return nn_Xs, nn_y


@app.cell
def _(nn_Xs, nn_y):
    import numpy as _np

    _rng = _np.random.default_rng(0)
    _n, _d = nn_Xs.shape
    _h = 16

    W1h = _rng.normal(scale=_np.sqrt(2 / (_d + _h)), size=(_d, _h))
    b1h = _np.zeros(_h)
    W2h = _rng.normal(scale=_np.sqrt(2 / (_h + 1)), size=(_h, 1))
    b2h = _np.zeros(1)

    _lr, _iters = 0.5, 3000
    nn_losses = []
    for _i in range(_iters):
        # forward pass
        _Z1 = nn_Xs @ W1h + b1h
        _A1 = _np.tanh(_Z1)
        _Z2 = _A1 @ W2h + b2h
        _P = 1.0 / (1.0 + _np.exp(-_np.clip(_Z2, -30, 30)))

        # backward pass (chain rule)
        _delta2 = (_P - nn_y.reshape(-1, 1)) / _n
        _dW2 = _A1.T @ _delta2
        _db2 = _delta2.sum(axis=0)
        _delta1 = (_delta2 @ W2h.T) * (1 - _A1**2)
        _dW1 = nn_Xs.T @ _delta1
        _db1 = _delta1.sum(axis=0)

        # gradient-descent update
        W2h -= _lr * _dW2
        b2h -= _lr * _db2
        W1h -= _lr * _dW1
        b1h -= _lr * _db1

        if _i % 100 == 0:
            _p = _P.ravel()
            nn_losses.append(
                -_np.mean(nn_y * _np.log(_p + 1e-12) + (1 - nn_y) * _np.log(1 - _p + 1e-12))
            )

    nn_acc = _np.mean((_P.ravel() >= 0.5) == nn_y)
    print(f"hand-written MLP (2 -> 16 -> 1): accuracy = {nn_acc:.3f}")
    return W1h, W2h, b1h, b2h, nn_acc, nn_losses


@app.cell
def _(b1h, b2h, mo, nn_Xs, nn_acc, nn_losses, nn_y, W1h, W2h):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np

    _xx, _yy = _np.meshgrid(
        _np.linspace(nn_Xs[:, 0].min() - 0.5, nn_Xs[:, 0].max() + 0.5, 250),
        _np.linspace(nn_Xs[:, 1].min() - 0.5, nn_Xs[:, 1].max() + 0.5, 250),
    )
    _grid = _np.c_[_xx.ravel(), _yy.ravel()]
    _P = 1.0 / (1.0 + _np.exp(-(_np.tanh(_grid @ W1h + b1h) @ W2h + b2h)))
    _Z = _P.reshape(_xx.shape)

    _fig, _axes = _plt.subplots(1, 2, figsize=(11, 4.2))
    _axes[0].plot(_np.arange(len(nn_losses)) * 100, nn_losses, color="#2563eb")
    _axes[0].set_xlabel("epoch")
    _axes[0].set_title("Cross-entropy during training")
    _axes[1].contourf(_xx, _yy, _Z, levels=20, cmap="RdYlGn", alpha=0.7)
    _axes[1].scatter(nn_Xs[nn_y == 0, 0], nn_Xs[nn_y == 0, 1], color="#dc2626", s=14)
    _axes[1].scatter(nn_Xs[nn_y == 1, 0], nn_Xs[nn_y == 1, 1], color="#16a34a", s=14)
    _axes[1].set_title(f"Learned boundary (by hand, acc {nn_acc:.2f})")
    _axes[1].set_aspect("equal")
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.image(_buf, width="860px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">9 / 14</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Same thing in scikit-learn

        ```python
        from sklearn.neural_network import MLPClassifier
        mlp = MLPClassifier(hidden_layer_sizes=(16,), activation="tanh",
                            max_iter=2000, random_state=0).fit(X, y)
        ```

        `MLPClassifier` uses **Adam** (a smarter gradient-descent variant),
        tracks its loss curve, and supports early stopping. Same network, same
        data — a more robust optimizer under the hood.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">10 / 14</div>
        """
    )
    return


@app.cell
def _(mo, nn_Xs, nn_y):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.neural_network import MLPClassifier as _MLP

    _mlp = _MLP(
        hidden_layer_sizes=(16,), activation="tanh",
        solver="adam", max_iter=1500, random_state=0,
    ).fit(nn_Xs, nn_y)

    _xx, _yy = _np.meshgrid(
        _np.linspace(nn_Xs[:, 0].min() - 0.5, nn_Xs[:, 0].max() + 0.5, 250),
        _np.linspace(nn_Xs[:, 1].min() - 0.5, nn_Xs[:, 1].max() + 0.5, 250),
    )
    _Z = _mlp.predict(_np.c_[_xx.ravel(), _yy.ravel()]).reshape(_xx.shape)

    _fig, _axes = _plt.subplots(1, 2, figsize=(11, 4.2))
    _axes[0].contourf(_xx, _yy, _Z, levels=20, cmap="RdYlGn", alpha=0.7)
    _axes[0].scatter(nn_Xs[nn_y == 0, 0], nn_Xs[nn_y == 0, 1], color="#dc2626", s=14)
    _axes[0].scatter(nn_Xs[nn_y == 1, 0], nn_Xs[nn_y == 1, 1], color="#16a34a", s=14)
    _axes[0].set_title(f"MLPClassifier — accuracy {_mlp.score(nn_Xs, nn_y):.2f}")
    _axes[0].set_aspect("equal")
    _axes[1].plot(_mlp.loss_curve_, color="#dc2626")
    _axes[1].set_xlabel("iteration")
    _axes[1].set_title("sklearn's loss curve (Adam)")
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.image(_buf, width="860px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">11 / 14</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Summary

        - **Forward pass** computes the activations; **backward pass**
          (backpropagation) applies the **chain rule** to get every gradient.
        - **Optimization** turns gradients into updates — the **learning rate**
          is the crucial knob: too small crawls, too large diverges.
        - We built an MLP **from scratch** in ~30 lines and it drew a round
          boundary no linear model could; sklearn does the same with a better
          optimizer (**Adam**).

        Next session: the **components** — activation functions, optimizers and
        weight initialization — and where neural networks go next.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">12 / 14</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Where to go next

        - **Exercise:** `notebooks/02/03_training_{beginner,intermediate,advanced}.ipynb`
          — implement the **forward and backward pass by hand**, train a small
          MLP, and compare against `MLPClassifier`.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">13 / 14</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Thanks for this session!

        Questions? Next up: **components and going beyond**.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">14 / 14</div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
