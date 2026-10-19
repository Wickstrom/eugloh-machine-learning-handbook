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
# Lecture 2 (Oct 20), Session 2 — Multilayer networks: forward pass, backward
# pass and training. One coherent story from "why stack neurons" through the
# MLP architecture, the forward pass, the backward pass (backpropagation),
# gradient descent, a from-scratch MLP, and the practical training components
# (initialization, optimizers, regularization, capacity). Multi-class softmax
# closes the session. Logistic regression and the sigmoid were Session 1.
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

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">1 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Today's session

        **Multilayer networks**
        - The **big idea** — stacking neurons to learn a useful transformation
        - The **MLP** — layers, shapes and matrix products
        - The **forward pass** — and what a hidden layer does to the data

        **Training a network**
        - The **cross-entropy** loss and the **backward pass** (backpropagation)
        - **Gradient descent**, the **learning rate**, and an MLP **from scratch**
        - Weight initialization, **optimizers**, regularization and **capacity**

        **Outputs**
        - **Multi-class** classification with **softmax**

        Session 2 of 4 today. (The sigmoid, activation functions and the XOR
        problem were covered in Session 1.)

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">2 / 20</div>
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

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 20</div>
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

        That chain is **matrix multiplication** — the reason GPUs are so good
        at this. (One hidden layer with enough units can approximate *any*
        continuous function — but nobody tells you how many units; you learn
        that from the data.)

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 20</div>
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

        - Each line is a **weighted sum (+ bias)** followed by the
          **activation** — exactly the single neuron from Session 1.
        - A whole batch is one matrix product: stack the inputs into $X$ and
          compute $Z^{(1)} = X W_1 + b_1$.
        - In plain words: **signals flow forward**, layer by layer, from inputs
          to prediction.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 20</div>
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
        input space: the network's boundary has to bend, so no single line
        works. On the right, the *same* points after one hidden layer
        ($\phi = \tanh$): they have been **pulled apart into a linearly
        separable arrangement**.

        Drag the sliders to move a point through the network and watch its
        hidden activation — that is the **forward pass**. (The network is
        trained on standardised inputs internally; the picture shows the raw
        measurements, so the sliders range over the unit square.)

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">6 / 20</div>
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
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">7 / 20</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Part 2 — Training a network

        ## Training a network — loss and gradients

        The recipe carries over from logistic regression: pick a **loss**, then
        descend its **gradient**. For classification the **cross-entropy** is

        $$\mathcal{L} = -\frac{1}{n}\sum_{i}\Big[\, y_i \log \hat p_i
        + (1 - y_i)\log(1 - \hat p_i)\,\Big]$$

        - small when the prediction matches the true label, large when it is
          **confidently wrong**;
        - for a *single* neuron it is **convex** — for a whole network it is
          **not**, so we settle for a good local minimum;
        - and, crucially, its gradient is a **simple** function of the error
          $\hat p - y$.

        What is new is computing that gradient *through* several layers — the
        **backward pass**.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">8 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The backward pass — backpropagation

        Work **backwards** from the output, applying the **chain rule** layer
        by layer:

        - **Output layer** — exactly like logistic regression: the "error" is
          $\hat p - y$.
        - **Hidden layer** — push that error back through the output weights,
          then multiply by the derivative of the activation. (For $\tanh$ that
          derivative is simply $1 - a^2$, where $a$ is the activation.)
        - **Repeat** — each layer's gradient is built from the *next* layer's
          error, so every step is only local arithmetic.

        The recipe:

        1. **Initialise** the parameters (small random weights)
        2. **Forward pass** — compute all activations
        3. **Backward pass** — gradients with the chain rule
        4. **Update** all parameters with gradient descent
        5. **Repeat!**

        In plain words: **blame flows backwards** — each weight is told how
        much it contributed to the final error. There is only **one loop** (the
        epochs); everything inside is matrix multiplication.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">9 / 20</div>
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

    _fig, _axes = _plt.subplots(1, 2, figsize=(11, 4))
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

    _text = mo.md(
        r"""
        ## Optimization — how we actually descend

        Backpropagation gives us the **gradients**; **optimization** decides how
        to use them:

        $$w \leftarrow w - \eta\,\frac{\partial \mathcal{L}}{\partial w}$$

        - $\eta$, the **learning rate**, is *the* single most important
          hyperparameter — too small crawls, too large **diverges**.
        - one pass over the whole training set is an **epoch**; batch /
          mini-batch / stochastic gradient descent trade step quality against
          cost — mini-batches are the workhorse of deep learning.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">10 / 20</div>
        """
    )
    mo.hstack(
        [_text, mo.image(_buf, width="470px")],
        widths=[1, 1], align="start", gap=1.5,
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.datasets import make_circles as _make_circles
    from sklearn.preprocessing import StandardScaler as _Scaler

    _X, _y = _make_circles(n_samples=400, factor=0.4, noise=0.1, random_state=0)
    _Xs = _Scaler().fit_transform(_X)
    _rng = _np.random.default_rng(0)
    _n, _d = _Xs.shape
    _h = 16

    _W1 = _rng.normal(scale=_np.sqrt(2 / (_d + _h)), size=(_d, _h))
    _b1 = _np.zeros(_h)
    _W2 = _rng.normal(scale=_np.sqrt(2 / (_h + 1)), size=(_h, 1))
    _b2 = _np.zeros(1)

    _lr, _iters = 0.5, 3000
    _losses = []
    for _i in range(_iters):
        # forward pass
        _Z1 = _Xs @ _W1 + _b1
        _A1 = _np.tanh(_Z1)
        _Z2 = _A1 @ _W2 + _b2
        _P = 1.0 / (1.0 + _np.exp(-_np.clip(_Z2, -30, 30)))

        # backward pass (chain rule)
        _delta2 = (_P - _y.reshape(-1, 1)) / _n
        _dW2 = _A1.T @ _delta2
        _db2 = _delta2.sum(axis=0)
        _delta1 = (_delta2 @ _W2.T) * (1 - _A1**2)
        _dW1 = _Xs.T @ _delta1
        _db1 = _delta1.sum(axis=0)

        # gradient-descent update
        _W2 -= _lr * _dW2
        _b2 -= _lr * _db2
        _W1 -= _lr * _dW1
        _b1 -= _lr * _db1

        if _i % 100 == 0:
            _p = _P.ravel()
            _losses.append(
                -_np.mean(_y * _np.log(_p + 1e-12) + (1 - _y) * _np.log(1 - _p + 1e-12))
            )

    _acc = _np.mean((_P.ravel() >= 0.5) == _y)

    _xx, _yy = _np.meshgrid(
        _np.linspace(_Xs[:, 0].min() - 0.5, _Xs[:, 0].max() + 0.5, 250),
        _np.linspace(_Xs[:, 1].min() - 0.5, _Xs[:, 1].max() + 0.5, 250),
    )
    _grid = _np.c_[_xx.ravel(), _yy.ravel()]
    _Zg = (1.0 / (1.0 + _np.exp(-(_np.tanh(_grid @ _W1 + _b1) @ _W2 + _b2)))).reshape(_xx.shape)

    _fig, _axes = _plt.subplots(1, 2, figsize=(11, 4.2))
    _axes[0].plot(_np.arange(len(_losses)) * 100, _losses, color="#2563eb")
    _axes[0].set_xlabel("epoch")
    _axes[0].set_title("Cross-entropy during training")
    _axes[1].contourf(_xx, _yy, _Zg, levels=20, cmap="RdYlGn", alpha=0.7)
    _axes[1].scatter(_Xs[_y == 0, 0], _Xs[_y == 0, 1], color="#dc2626", s=14)
    _axes[1].scatter(_Xs[_y == 1, 0], _Xs[_y == 1, 1], color="#16a34a", s=14)
    _axes[1].set_title(f"Learned boundary (by hand, acc {_acc:.2f})")
    _axes[1].set_aspect("equal")
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)

    _text = mo.md(
        r"""
        ## An MLP by hand, in numpy

        2 inputs → 16 hidden units ($\tanh$) → 1 output (sigmoid), trained with
        **full-batch gradient descent** on the circles dataset (standardised).
        Forward pass, backward pass, update — a handful of lines:

        ```python
        for _ in range(iters):
            A1 = tanh(X @ W1 + b1);  P = sigmoid(A1 @ W2 + b2)   # forward
            d2 = (P - y) / n;  dW2 = A1.T @ d2                   # backward
            d1 = (d2 @ W2.T) * (1 - A1**2);  dW1 = X.T @ d1
            W2 -= lr * dW2;  W1 -= lr * dW1                       # update
        ```

        It drew a **round** boundary that no linear model could — exactly the
        shape the data demands.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">11 / 20</div>
        """
    )
    mo.hstack(
        [_text, mo.image(_buf, width="470px")],
        widths=[1, 1], align="start", gap=1.5,
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The same in scikit-learn

        ```python
        from sklearn.neural_network import MLPClassifier
        mlp = MLPClassifier(hidden_layer_sizes=(16,), activation="tanh",
                            max_iter=2000, random_state=0).fit(X, y)
        ```

        Same network, same data — but scikit-learn swaps our plain gradient
        descent for **Adam** (a smarter variant), and adds **early stopping**
        and **regularization** out of the box. Everything we built by hand is
        one line.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">12 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np

    _rng = _np.random.default_rng(0)
    _n_layers, _width = 12, 100
    _he = _np.sqrt(2 / _width)
    _scales = {
        "too small (0.1·He)": 0.1 * _he,
        "He init (good)": _he,
        "too large (5·He)": 5 * _he,
    }
    _colors = ["#2563eb", "#16a34a", "#dc2626"]

    _fig, _ax = _plt.subplots(figsize=(6.6, 4.2))
    for (_name, _s), _c in zip(_scales.items(), _colors):
        _a = _rng.normal(size=(1000, _width))
        _stds = [_a.std()]
        for _ in range(_n_layers):
            _W = _rng.normal(scale=_s, size=(_width, _width))
            _a = _np.maximum(0, _a @ _W)  # ReLU
            _stds.append(_a.std())
        _ax.plot(range(_n_layers + 1), _stds, "o-", color=_c, label=_name)

    _ax.set_yscale("log")
    _ax.set_xlabel("layer")
    _ax.set_ylabel("activation std (log scale)")
    _ax.set_title("Weight scale controls signal through the layers")
    _ax.legend(fontsize=8)
    _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)

    _text = mo.md(
        r"""
        # Part 3 — Training in practice

        ## Weight initialization

        You cannot start all weights at **zero** — every neuron would be
        identical and stay identical (no **symmetry breaking**). But *how* we
        randomise matters:

        - **too small** → activations shrink layer by layer → **vanishing**
          signal, no learning;
        - **too large** → activations blow up → **exploding** signal.

        The fix is to scale the weights by the layer's **fan-in** — **He
        initialisation** sets the standard deviation to
        $\sigma = \sqrt{2 / n_{\text{in}}}$ (Xavier is similar, for $\tanh$).

        scikit-learn does this for you; in PyTorch it is `nn.init.kaiming_normal_`.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">13 / 20</div>
        """
    )
    mo.hstack(
        [_text, mo.image(_buf, width="450px")],
        widths=[1, 1], align="start", gap=1.5,
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt
    from sklearn.datasets import make_moons as _make_moons
    from sklearn.model_selection import train_test_split as _split
    from sklearn.neural_network import MLPClassifier as _MLP
    from sklearn.preprocessing import StandardScaler as _Scaler

    _X, _y = _make_moons(n_samples=400, noise=0.3, random_state=0)
    _Xtr, _Xte, _ytr, _yte = _split(_X, _y, test_size=0.3, random_state=0, stratify=_y)
    _sc = _Scaler().fit(_Xtr)
    _Xtr, _Xte = _sc.transform(_Xtr), _sc.transform(_Xte)

    _configs = [
        ("SGD", {"solver": "sgd", "momentum": 0.0, "learning_rate_init": 0.1}, "#2563eb"),
        ("SGD + momentum", {"solver": "sgd", "momentum": 0.9, "learning_rate_init": 0.05}, "#16a34a"),
        ("Adam", {"solver": "adam"}, "#dc2626"),
    ]

    _fig, _ax = _plt.subplots(figsize=(6.6, 4.2))
    for _name, _kw, _c in _configs:
        _m = _MLP(
            hidden_layer_sizes=(32,), max_iter=2000, random_state=0, **_kw
        ).fit(_Xtr, _ytr)
        _ax.plot(_m.loss_curve_, color=_c,
                 label=f"{_name} (test {_m.score(_Xte, _yte):.2f})")

    _ax.set_xlabel("iteration")
    _ax.set_ylabel("loss")
    _ax.set_title("Optimizers compared on the two-moons task")
    _ax.legend(fontsize=8)
    _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)

    _text = mo.md(
        r"""
        ## Optimizers — beyond plain gradient descent

        Plain gradient descent is the starting point; three famous upgrades:

        - **SGD** — estimate the gradient on a **mini-batch** instead of the
          whole dataset: noisier, far cheaper per step.
        - **Momentum** — carry part of the previous step forward, like a ball
          rolling downhill: $v \leftarrow \beta v - \eta\nabla\mathcal{L}$,
          then $w \leftarrow w + v$.
        - **Adam** — the default in practice. It tracks a running average of
          the gradient and of its magnitude, adapting the learning rate
          **per weight** — forgiving of a badly chosen $\eta$.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">14 / 20</div>
        """
    )
    mo.hstack(
        [_text, mo.image(_buf, width="450px")],
        widths=[1, 1], align="start", gap=1.5,
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.datasets import make_moons as _make_moons
    from sklearn.model_selection import train_test_split as _split
    from sklearn.neural_network import MLPClassifier as _MLP
    from sklearn.preprocessing import StandardScaler as _Scaler

    _X, _y = _make_moons(n_samples=400, noise=0.3, random_state=0)
    _Xtr, _Xte, _ytr, _yte = _split(
        _X, _y, test_size=0.3, random_state=0, stratify=_y
    )
    _sc = _Scaler().fit(_Xtr)
    _Xtr, _Xte = _sc.transform(_Xtr), _sc.transform(_Xte)

    _alphas = _np.logspace(-6, -1, 8)
    _tr_acc, _te_acc = [], []
    for _a in _alphas:
        _m = _MLP(
            hidden_layer_sizes=(32,), alpha=_a, max_iter=1500, random_state=0
        ).fit(_Xtr, _ytr)
        _tr_acc.append(_m.score(_Xtr, _ytr))
        _te_acc.append(_m.score(_Xte, _yte))

    _fig, _ax = _plt.subplots(figsize=(6.6, 4.2))
    _ax.semilogx(_alphas, _tr_acc, "o-", color="#dc2626", label="train")
    _ax.semilogx(_alphas, _te_acc, "o-", color="#16a34a", label="test")
    _ax.set_xlabel("alpha (L2 strength)")
    _ax.set_ylabel("accuracy")
    _ax.set_title("Weight decay: a little train accuracy buys a lot of test")
    _ax.legend(fontsize=8)
    _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)

    _text = mo.md(
        r"""
        ## Regularization — keeping the network honest

        A flexible network can **memorise** the training set. Regularization
        adds a cost for complexity so the model prefers simpler solutions:

        - **Weight decay (L2)** — penalise large weights;
          `MLPClassifier(alpha=...)` (larger = stronger), the same penalty as
          `Ridge`.
        - **Dropout** — randomly switch off neurons during training so the
          network cannot rely on any single one.
        - **Early stopping** — stop when the **validation** loss stops
          improving: `MLPClassifier(early_stopping=True)`.

        You trade a little **training** accuracy for a lot of **test**
        accuracy.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">15 / 20</div>
        """
    )
    mo.hstack(
        [_text, mo.image(_buf, width="450px")],
        widths=[1, 1], align="start", gap=1.5,
    )
    return


@app.cell
def _(mo):
    hidden_slider = mo.ui.slider(
        start=1, stop=32, step=1, value=8,
        label="hidden units", show_value=True, debounce=True,
    )
    act_dropdown = mo.ui.dropdown(
        options=["tanh", "relu", "logistic"], value="tanh",
        label="activation",
    )
    mo.md(
        r"""
        ## Capacity — width and depth

        - **Width** — more units in a hidden layer.
        - **Depth** — more hidden layers, each building on the previous one
          (edges → parts → objects, in vision).

        More capacity fits more complex boundaries — and **overfits** sooner.
        Drag the slider and swap the activation: a single hidden unit cannot
        even bend the boundary.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">16 / 20</div>
        """
    )
    return act_dropdown, hidden_slider


@app.cell
def _(act_dropdown, hidden_slider, mo):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.datasets import make_circles as _make_circles
    from sklearn.neural_network import MLPClassifier as _MLP
    from sklearn.preprocessing import StandardScaler as _Scaler

    _X, _y = _make_circles(n_samples=400, factor=0.4, noise=0.1, random_state=0)
    _Xs = _Scaler().fit_transform(_X)

    _mlp = _MLP(
        hidden_layer_sizes=(hidden_slider.value,),
        activation=act_dropdown.value,
        max_iter=1500,
        random_state=0,
    ).fit(_Xs, _y)

    _xx, _yy = _np.meshgrid(
        _np.linspace(_Xs[:, 0].min() - 0.5, _Xs[:, 0].max() + 0.5, 250),
        _np.linspace(_Xs[:, 1].min() - 0.5, _Xs[:, 1].max() + 0.5, 250),
    )
    _Z = _mlp.predict(_np.c_[_xx.ravel(), _yy.ravel()]).reshape(_xx.shape)

    _fig, _ax = _plt.subplots(figsize=(5.4, 4.6))
    _ax.contourf(_xx, _yy, _Z, alpha=0.25, cmap="RdYlGn")
    _ax.scatter(_Xs[_y == 0, 0], _Xs[_y == 0, 1], color="#dc2626", s=14)
    _ax.scatter(_Xs[_y == 1, 0], _Xs[_y == 1, 1], color="#16a34a", s=14)
    _ax.set_title(
        f"({hidden_slider.value},) {act_dropdown.value} · "
        f"accuracy {_mlp.score(_Xs, _y):.2f}"
    )
    _ax.set_aspect("equal")
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.hstack([hidden_slider, act_dropdown]),
            mo.image(_buf, width="620px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">17 / 20</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Training in practice

        Getting a network to train is a **craft**. The reliable recipe:

        1. **Split first** — train / validation / test. Never tune on the test set.
        2. **Scale the inputs** — networks are very sensitive to feature scales.
        3. **Start small** — a tiny network, and a baseline (`LogisticRegression`).
        4. **Watch both curves** — training loss *and* validation loss.
        5. **Change one thing at a time** — learning rate, width, regularization.
        6. Only then **scale up** — more data, more depth, more epochs.

        | symptom | likely cause | fix |
        |---|---|---|
        | loss barely moves | lr too small, unscaled features | raise lr, standardise |
        | loss explodes / `NaN` | lr too large | lower it ~10×, standardise |
        | train great, validation poor | **overfitting** | more data, `alpha`, dropout, early stopping |
        | both poor | **underfitting** | bigger network, train longer |
        | stuck at the majority class | class imbalance | class weights, resampling, better metric |

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">18 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt
    from sklearn.datasets import load_digits as _load_digits
    from sklearn.metrics import ConfusionMatrixDisplay as _CMD
    from sklearn.model_selection import train_test_split as _split
    from sklearn.neural_network import MLPClassifier as _MLP

    _digits = _load_digits()
    _X = _digits.data / 16.0
    _y = _digits.target
    _Xtr, _Xte, _ytr, _yte = _split(_X, _y, test_size=0.3, random_state=0, stratify=_y)

    _mlp = _MLP(hidden_layer_sizes=(24,), max_iter=400, random_state=0).fit(_Xtr, _ytr)
    _acc = _mlp.score(_Xte, _yte)

    _fig, _axes = _plt.subplots(1, 2, figsize=(11, 4.0),
                                gridspec_kw={"width_ratios": [1, 1.4]})
    _axes[0].imshow(_digits.images[0], cmap="gray_r")
    _axes[0].set_title(f"an 8×8 digit — test accuracy {_acc:.3f}")
    _axes[0].set_xticks([])
    _axes[0].set_yticks([])
    _CMD.from_predictions(_yte, _mlp.predict(_Xte), ax=_axes[1], colorbar=False)
    _axes[1].set_title("Confusion matrix — which digits get confused?")
    _fig.tight_layout()
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)

    _text = mo.md(
        r"""
        # Part 4 — Outputs

        ## Multi-class: softmax output

        With $K$ classes the last layer has $K$ neurons; **softmax** turns their
        raw scores into a probability distribution:

        $$\hat{p}_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}},
        \qquad \sum_k \hat{p}_k = 1$$

        Exponentiate each score (to make it positive), then divide by the
        total. The loss is the multi-class cross-entropy — `MLPClassifier`
        handles all of it. Demo: 8×8 handwritten digits, 10 classes.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">19 / 20</div>
        """
    )
    mo.vstack([_text, mo.image(_buf, width="860px")])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Summary

        - A single neuron is **linear**; a **hidden layer** transforms the
          inputs so the last layer can separate them — that is the **MLP**.
        - The **forward pass** sends the signal through the network; the
          **backward pass** (backpropagation) uses the **chain rule** to get
          every gradient.
        - **Gradient descent** turns gradients into updates; the **learning
          rate** is the crucial knob. We built the whole thing **from scratch**
          in a few lines.
        - Training in practice: **initialization**, **optimizers**, **regularization**
          and **capacity** are the levers — and we watch the curves.
        - Many classes use the **softmax** output layer.

        Next session: **beyond the MLP** — convolutional networks, transformers
        and autoencoders.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">20 / 20</div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
