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
# Opens with the regression-to-classification bridge and "logistic regression is
# one neuron" (the sigmoid itself is introduced in Session 1); then the big idea
# (stacking neurons), what a hidden layer does to the data, the forward pass, the
# multi-layer perceptron, and multi-class classification.
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

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">1 / 16</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Today's session

        - **From regression to classification** — why a straight line will not do,
          and why **logistic regression is one neuron**
        - **The big idea** — stacking neurons to learn a useful transformation
        - **The multi-layer perceptron (MLP)** — layers, shapes, matrix products
        - **The forward pass** — computing activations layer by layer
        - **What a hidden layer does** — reshaping the data until it is separable
        - **Multi-class classification** — one-vs-rest and softmax

        Session 2 of 4 today. (The sigmoid, activation functions and the XOR
        problem were covered in Session 1.)

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">2 / 16</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## From regression to classification

        Linear regression predicts a **number**. But many problems ask for a
        **label** (spam / not spam, healthy / diseased).

        Feed $0/1$ labels into a line and it happily predicts $-0.4$ or $1.4$ —
        which is meaningless as a probability:

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 16</div>
        """
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.linear_model import LinearRegression as _LinReg

    _rng = _np.random.default_rng(0)
    _x0 = _rng.normal(loc=2.0, scale=0.9, size=60)
    _x1 = _rng.normal(loc=6.0, scale=0.9, size=60)
    _x = _np.concatenate([_x0, _x1])
    _y = _np.concatenate([_np.zeros(60), _np.ones(60)])

    _line = _LinReg().fit(_x.reshape(-1, 1), _y)
    _xs = _np.linspace(0, 8, 100)
    _yh = _line.predict(_xs.reshape(-1, 1))

    _fig, _ax = _plt.subplots(figsize=(7, 4))
    _ax.scatter(_x[_y == 0], _y[_y == 0], color="#dc2626", s=30, label="class 0")
    _ax.scatter(_x[_y == 1], _y[_y == 1], color="#16a34a", s=30, label="class 1")
    _ax.plot(_xs, _yh, color="#2563eb", lw=2, label="linear regression fit")
    _ax.axhline(0, color="gray", ls="--", lw=0.8)
    _ax.axhline(1, color="gray", ls="--", lw=0.8)
    _ax.set_xlabel("feature $x$")
    _ax.set_ylabel("label $y$")
    _ax.set_title("Labels are 0/1 — the line predicts −0.4 … 1.4. Not adapted!")
    _ax.legend(loc="center left")
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.image(_buf, width="620px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 16</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Logistic regression is one neuron!

        A **neuron** computes a weighted sum and applies a non-linear
        activation $\phi$:

        $$z = w x + b, \qquad \hat{p} = \phi(z) \quad \text{with } \phi = \sigma$$

        Logistic regression **is** exactly one neuron with a sigmoid
        activation. Training is parametrised by a **learning rate** $\eta$;
        we show the full dataset multiple times — each complete pass is one
        **epoch**.

        Keep this picture — the next slides stack these neurons into a network.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 16</div>
        """
    )
    return


@app.cell
def _(mo):
    import numpy as _np

    _rng = _np.random.default_rng(3)
    _Xa = _rng.normal(loc=[-2.0, -2.0], scale=1.0, size=(20, 2))
    _Xb = _rng.normal(loc=[2.0, 2.0], scale=1.0, size=(20, 2))
    incL_X = _np.empty((40, 2))
    incL_y = _np.empty(40, dtype=int)
    for _i in range(20):  # interleave the classes so every prefix has both
        incL_X[2 * _i] = _Xa[_i]
        incL_y[2 * _i] = 0
        incL_X[2 * _i + 1] = _Xb[_i]
        incL_y[2 * _i + 1] = 1
    logit_seen = mo.ui.slider(
        start=4, stop=40, step=2, value=8,
        label="samples seen", show_value=True, debounce=True,
    )
    mo.md(
        r"""
        ## Interactive — the boundary as the data arrives

        The same story, now for classification. The slider reveals the first
        $k$ samples and we re-fit the logistic model on those $k$ points
        alone.

        - early on, one or two points can tilt the **decision boundary** a lot
        - as $k$ grows the boundary **stabilises** and the true separation
          emerges
        - the boundary is where the model is exactly $50/50$ ($z = 0$)

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">6 / 16</div>
        """
    )
    return incL_X, incL_y, logit_seen


@app.cell
def _(incL_X, incL_y, logit_seen, mo):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.linear_model import LogisticRegression as _LR

    _k = int(logit_seen.value)
    _Xk, _yk = incL_X[:_k], incL_y[:_k]
    _clf = _LR().fit(_Xk, _yk)
    _w, _b = _clf.coef_[0], _clf.intercept_[0]

    _xx, _yy = _np.meshgrid(_np.linspace(-6, 6, 220), _np.linspace(-6, 6, 220))
    _Z = _clf.predict(_np.c_[_xx.ravel(), _yy.ravel()]).reshape(_xx.shape)

    _fig, _ax = _plt.subplots(figsize=(6.0, 5.0))
    _ax.contourf(_xx, _yy, _Z, alpha=0.2, cmap="RdYlGn")
    _xs_line = _np.linspace(-6, 6, 100)
    if abs(_w[1]) > 1e-9:
        _ax.plot(_xs_line, -(_w[0] * _xs_line + _b) / _w[1],
                 color="#111827", lw=2.2, label="decision boundary")
    _ax.scatter(incL_X[_k:, 0], incL_X[_k:, 1], color="#cbd5e1", s=26,
                label="not yet seen")
    _ax.scatter(_Xk[_yk == 0, 0], _Xk[_yk == 0, 1], color="#dc2626", s=34,
                label="seen, class 0")
    _ax.scatter(_Xk[_yk == 1, 0], _Xk[_yk == 1, 1], color="#16a34a", s=34,
                label="seen, class 1")
    _ax.set_xlim(-6, 6)
    _ax.set_ylim(-6, 6)
    _ax.set_xlabel("$x_1$")
    _ax.set_ylabel("$x_2$")
    _ax.set_title(f"{_k} samples → accuracy {_clf.score(_Xk, _yk):.2f}")
    _ax.legend(loc="upper left", fontsize=8)
    _ax.set_aspect("equal")
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            logit_seen,
            mo.image(_buf, width="600px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">7 / 16</div>"""),
        ]
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

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">8 / 16</div>
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

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">9 / 16</div>
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

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">10 / 16</div>
        """
    )
    return


@app.cell
def _(mo):
    x1_slider = mo.ui.slider(
        start=-0.5, stop=1.5, step=0.05, value=1.0,
        label="input $x_1$", show_value=True, debounce=True,
    )
    x2_slider = mo.ui.slider(
        start=-0.5, stop=1.5, step=0.05, value=0.0,
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
        hidden activation — that is the **forward pass**. (The network is
        trained on standardised inputs internally; the picture shows the raw
        measurements, so the sliders range over the unit square.)

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">11 / 16</div>
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

    # forward pass for the point chosen by the sliders (raw coordinates);
    # the network expects standardised inputs, so transform the query first.
    _q = _np.array([[x1_slider.value, x2_slider.value]])
    _qs = _sc.transform(_q)
    _qa = _np.tanh(_qs @ _mlp.coefs_[0] + _mlp.intercepts_[0])
    _prob = _mlp.predict_proba(_qs)[0, 1]

    _xx, _yy = _np.meshgrid(
        _np.linspace(_X[:, 0].min() - 0.15, _X[:, 0].max() + 0.15, 220),
        _np.linspace(_X[:, 1].min() - 0.15, _X[:, 1].max() + 0.15, 220),
    )
    _Zin = _mlp.predict(
        _sc.transform(_np.c_[_xx.ravel(), _yy.ravel()])
    ).reshape(_xx.shape)

    _hx, _hy = _np.meshgrid(
        _np.linspace(_A1[:, 0].min() - 0.3, _A1[:, 0].max() + 0.3, 220),
        _np.linspace(_A1[:, 1].min() - 0.3, _A1[:, 1].max() + 0.3, 220),
    )
    _Zh = _lr.predict(_np.c_[_hx.ravel(), _hy.ravel()]).reshape(_hx.shape)

    _fig, _axes = _plt.subplots(1, 2, figsize=(11, 4.4))
    _axes[0].contourf(_xx, _yy, _Zin, alpha=0.22, cmap="RdYlGn")
    _axes[0].scatter(_X[_y == 0, 0], _X[_y == 0, 1], color="#dc2626", s=16)
    _axes[0].scatter(_X[_y == 1, 0], _X[_y == 1, 1], color="#16a34a", s=16)
    _axes[0].scatter([_q[0, 0]], [_q[0, 1]], marker="*", s=260,
                     color="#2563eb", edgecolor="k", zorder=5)
    _axes[0].set_xlabel("$x_1$")
    _axes[0].set_ylabel("$x_2$")
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
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">12 / 16</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Multi-class classification

        More than two classes? Two standard tricks:

        - **One-vs-rest** — train one binary classifier per class ("this class
          vs everything else"), predict the most confident one.
        - **Softmax / multinomial** — generalise the sigmoid: a vector of
          probabilities, one per class, summing to 1.

        scikit-learn's `LogisticRegression` handles this automatically.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">13 / 16</div>
        """
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.datasets import load_iris as _load_iris
    from sklearn.linear_model import LogisticRegression as _LR

    _iris = _load_iris()
    _X = _iris.data[:, 2:4]  # petal length & width
    _y = _iris.target
    _names = list(_iris.target_names)

    # One-vs-rest: a separate binary "this class vs all the others" model.
    _ovr = [
        _LR(max_iter=1000).fit(_X, (_y == _k).astype(int))
        for _k in range(len(_names))
    ]

    _x0, _x1 = _X[:, 0].min() - 0.5, _X[:, 0].max() + 0.5
    _y0, _y1 = _X[:, 1].min() - 0.5, _X[:, 1].max() + 0.5
    _xx, _yy = _np.meshgrid(_np.linspace(_x0, _x1, 300), _np.linspace(_y0, _y1, 300))
    _grid = _np.c_[_xx.ravel(), _yy.ravel()]
    _probs = _np.column_stack([_m.predict_proba(_grid)[:, 1] for _m in _ovr])
    _final = _np.argmax(_probs, axis=1).reshape(_xx.shape)

    # A 2x2 grid (rather than one long strip) keeps every panel large enough
    # to read. All four panels share the same axes, so the three "vs rest"
    # boundaries are directly comparable.
    _fig, _axes = _plt.subplots(2, 2, figsize=(11.5, 6.2))
    _axes = _axes.ravel()
    _xs = _np.linspace(_x0, _x1, 100)
    for _k in range(len(_names)):
        _ax = _axes[_k]
        _m = _ovr[_k]
        _w, _b = _m.coef_[0], _m.intercept_[0]
        # Shade the half-plane this classifier calls "the class" vs "rest".
        _mask = _m.predict(_grid).reshape(_xx.shape)
        _ax.contourf(_xx, _yy, _mask, levels=[-0.5, 0.5, 1.5],
                     colors=["#e2e8f0", "#86efac"], alpha=0.55)
        _ax.scatter(_X[_y != _k, 0], _X[_y != _k, 1], color="#64748b", s=22,
                    alpha=0.7, label="rest")
        _ax.scatter(_X[_y == _k, 0], _X[_y == _k, 1], color="#16a34a", s=26,
                    label=_names[_k])
        if abs(_w[1]) > 1e-9:
            _ax.plot(_xs, -(_w[0] * _xs + _b) / _w[1], color="#111827", lw=2.2)
        _acc_k = _m.score(_X, (_y == _k).astype(int))
        _ax.set_title(f"{_names[_k]} vs rest — binary accuracy {_acc_k:.2f}")
        _ax.set_xlim(_x0, _x1)
        _ax.set_ylim(_y0, _y1)
        _ax.set_aspect("equal")
        _ax.legend(fontsize=8, loc="upper left")
        if _k >= 2:
            _ax.set_xlabel("petal length (cm)")
        if _k % 2 == 0:
            _ax.set_ylabel("petal width (cm)")

    _axes[3].contourf(_xx, _yy, _final, alpha=0.30, cmap="RdYlGn")
    for _k in range(len(_names)):
        _axes[3].scatter(_X[_y == _k, 0], _X[_y == _k, 1], s=26,
                         label=_names[_k])
    _axes[3].set_title("final: most confident class")
    _axes[3].set_xlim(_x0, _x1)
    _axes[3].set_ylim(_y0, _y1)
    _axes[3].set_aspect("equal")
    _axes[3].legend(fontsize=8, loc="upper left")
    _axes[3].set_xlabel("petal length (cm)")

    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.md(
                r"""Each panel trains a **separate** binary classifier for
                *that class vs everything else* (green = the region it calls
                "the class"). Setosa and virginica separate cleanly, but the
                middle class **versicolor is not linearly separable** from the
                rest: its best straight line still misclassifies ~40% of the
                data. The last panel keeps whichever classifier is most
                confident."""
            ),
            mo.image(_buf, width="920px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">14 / 16</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Summary

        - **Logistic regression is one neuron** — a weighted sum plus a sigmoid;
          more than two classes use **one-vs-rest** or **softmax**.
        - A single linear unit cannot solve **XOR** — but a **hidden layer** can
          transform the inputs so the last layer separates them.
        - The **forward pass** sends the signal through the network:
          weighted sums and activations, layer by layer.
        - An **MLP** stacks neurons in layers; the whole computation is a
          sequence of **matrix products**, trained end-to-end by
          **backpropagation**.

        Next session: **how the network learns** — backpropagation, gradient
        descent and optimization.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">15 / 16</div>
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

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">16 / 16</div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
