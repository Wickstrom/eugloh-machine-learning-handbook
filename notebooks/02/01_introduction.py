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
# From linear to non-linear classifiers; the artificial neuron and activation
# functions; the XOR problem; and a brief history of neural networks.
# Run locally with `marimo edit notebooks/02/01_introduction.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).
#
# NOTE on scoping: Marimo requires each global name to be owned by exactly
# one cell. All cell-locals are underscore-prefixed.

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

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">1 / 13</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Today's session

        - **From linear to non-linear** classifiers — why we need them
        - **The artificial neuron** — a weighted sum plus a non-linear activation
        - **Activation functions** — the non-linearity that makes networks powerful
        - **The XOR problem** — the limitation of a single linear unit
        - A brief **history** of neural networks — and the two AI winters

        Session 1 of 4 today.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">2 / 13</div>
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

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 13</div>
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
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 13</div>"""),
        ]
    )
    return nn_X, nn_Xs, nn_y


@app.cell
def _(mo):
    mo.md(
        r"""
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

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 13</div>
        """
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.patches as _mpatches
    import matplotlib.pyplot as _plt
    import numpy as _np

    _fig, _ax = _plt.subplots(figsize=(7.8, 3.2))
    _ax.axis("off")
    _ax.set_xlim(0, 1)
    _ax.set_ylim(0, 1)

    _in_x = 0.10
    _in_y = [0.78, 0.5, 0.22]
    _labels = ["$x_1$", "$x_2$", "$x_3$"]
    _ws = ["$w_1$", "$w_2$", "$w_3$"]
    for _yy, _lab in zip(_in_y, _labels):
        _ax.add_patch(_plt.Circle((_in_x, _yy), 0.05, color="#2563eb", zorder=3))
        _ax.text(_in_x, _yy, _lab, ha="center", va="center", color="white", fontsize=11, zorder=4)

    _sx, _sy = 0.52, 0.5
    for _yy, _w in zip(_in_y, _ws):
        _ax.annotate("", xy=(_sx - 0.065, _sy), xytext=(_in_x + 0.05, _yy),
                     arrowprops=dict(arrowstyle="-|>", color="#6b7280", lw=1.4))
        _ax.text((_in_x + _sx) / 2, (_yy + _sy) / 2 + 0.035, _w,
                 color="#6b7280", fontsize=10, ha="center")
    _ax.add_patch(_plt.Circle((_sx, _sy), 0.065, color="#111827", zorder=3))
    _ax.text(_sx, _sy, r"$\Sigma$", ha="center", va="center", color="white", fontsize=15, zorder=4)
    _ax.annotate("", xy=(_sx, _sy - 0.065), xytext=(_sx, 0.12),
                 arrowprops=dict(arrowstyle="-|>", color="#6b7280", lw=1.4))
    _ax.text(_sx + 0.03, 0.12, "bias $b$", color="#6b7280", fontsize=10, va="center")
    _ax.text(_sx, _sy + 0.12, "weighted sum", ha="center", color="#111827", fontsize=9)

    _ax.annotate("", xy=(0.70, _sy), xytext=(_sx + 0.065, _sy),
                 arrowprops=dict(arrowstyle="-|>", color="#6b7280", lw=1.4))
    _ax.add_patch(_mpatches.FancyBboxPatch((0.70, 0.40), 0.13, 0.20,
                 boxstyle="round,pad=0.02", fc="#16a34a", ec="none", zorder=3))
    _ax.text(0.765, _sy, r"$\phi$", ha="center", va="center", color="white", fontsize=15, zorder=4)
    _ax.text(0.765, 0.32, "activation", ha="center", color="#16a34a", fontsize=9)

    _ax.annotate("", xy=(0.93, _sy), xytext=(0.83, _sy),
                 arrowprops=dict(arrowstyle="-|>", color="#6b7280", lw=1.4))
    _ax.add_patch(_plt.Circle((0.95, _sy), 0.05, color="#dc2626", zorder=3))
    _ax.text(0.95, _sy, "$a$", ha="center", va="center", color="white", fontsize=11, zorder=4)

    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.image(_buf, width="760px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">6 / 13</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Activation functions — the non-linearity

        | function | formula | range | notes |
        |---|---|---|---|
        | sigmoid | $\sigma(z) = 1/(1+e^{-z})$ | $(0,1)$ | probability-like; saturates → vanishing gradients |
        | tanh | $\tanh(z)$ | $(-1,1)$ | zero-centred; hidden-layer classic |
        | ReLU | $\max(0, z)$ | $[0,\infty)$ | cheap, no saturation for $z>0$ — today's default |
        | softmax | $e^{z_k}/\sum_j e^{z_j}$ | $(0,1)$, sums to 1 | output layer for multi-class |

        The non-linearity is the whole point: without it, stacking layers is
        still just a straight line. Compose enough of them and the network can
        approximate **any** smooth function.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">7 / 13</div>
        """
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np

    _z = _np.linspace(-4, 4, 300)
    _fig, _ax = _plt.subplots(figsize=(7, 3.6))
    _ax.plot(_z, 1 / (1 + _np.exp(-_z)), lw=2, label="sigmoid", color="#2563eb")
    _ax.plot(_z, _np.tanh(_z), lw=2, label="tanh", color="#16a34a")
    _ax.plot(_z, _np.maximum(0, _z), lw=2, label="ReLU", color="#dc2626")
    _ax.axhline(0, color="gray", lw=0.5)
    _ax.axvline(0, color="gray", lw=0.5)
    _ax.set_title("Activation functions")
    _ax.legend()
    _ax.grid(alpha=0.3)
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.image(_buf, width="620px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">8 / 13</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The XOR problem — one neuron is not enough

        A neuron draws a **single straight line** in input space. But some
        patterns simply cannot be split by a line — the classic example is
        **XOR**: the green class sits on the *diagonal*, the red class on the
        *anti-diagonal*.

        No straight line separates them, so **a single perceptron can never
        solve XOR**. In 1969 Minsky & Papert proved this rigorously — and it
        triggered the first **AI winter**. The fix (a **hidden layer**) is the
        subject of the next session.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">9 / 13</div>
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
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">10 / 13</div>"""),
        ]
    )
    return


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

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">11 / 13</div>
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
        - A **neuron** is a weighted sum plus a non-linear **activation**;
          **logistic regression is one neuron**.
        - The **activation function** supplies the non-linearity; without it a
          network collapses to a straight line.
        - A single neuron cannot solve **XOR** — the fix is to **stack neurons
          in layers**, which we build next.

        Next session: **multilayer networks** — how a hidden layer reshapes the
        data, and the **forward pass**.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">12 / 13</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Where to go next

        - **Exercise:** `notebooks/02/01_introduction_{beginner,intermediate,advanced}.ipynb`
          — visualise the non-linear boundaries, play with activation functions,
          and see why a single neuron cannot solve XOR.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">13 / 13</div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
