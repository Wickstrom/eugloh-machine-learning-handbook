# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "scikit-learn",
#     "scipy",
# ]
# ///
#
# Lecture 2 (Oct 20), Session 4 — Components (activation functions, optimizers,
# weight initialization, regularization, capacity) and going beyond (multi-class
# softmax, convolutional networks, transformers, autoencoders).
# Run locally with `marimo edit notebooks/02/04_components_and_beyond.py`
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
    layout_file="layouts/04_components_and_beyond.slides.json",
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
        # Components & Going Beyond

        **Machine Learning with Python** — Lecture 2, Session 4 (Oct 20)

        EUGLOH — *Problem Solving Using Open-Source Languages; R and Python*

        University of Novi Sad

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">1 / 24</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Today's session

        **Components of a neural network**
        - **Activation functions** — the non-linearity
        - **Weight initialization** — why it matters
        - **Optimizers** — SGD, momentum, Adam
        - **Regularization** and **capacity** — the bias–variance levers

        **Going beyond**
        - Multi-class with **softmax**, **convolutional networks**,
          **transformers**, **autoencoders** — and when to use which model

        Session 4 of 4 today — this wraps the lecture series.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">2 / 24</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Part 1 — Components

        ## Activation functions

        | function | formula | range | notes |
        |---|---|---|---|
        | sigmoid | $\sigma(z) = 1/(1+e^{-z})$ | $(0,1)$ | probability-like; saturates → vanishing gradients |
        | tanh | $\tanh(z)$ | $(-1,1)$ | zero-centred; hidden-layer classic |
        | ReLU | $\max(0, z)$ | $[0,\infty)$ | cheap, no saturation for $z>0$ — today's default |
        | softmax | $e^{z_k}/\sum_j e^{z_j}$ | $(0,1)$, sums to 1 | output layer for multi-class |

        The non-linearity is the whole point: compose enough of them and the
        network can approximate **any** smooth function.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 24</div>
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
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 24</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Weight initialization

        You cannot start all weights at zero — every neuron would be identical
        and stay identical (no symmetry breaking). But **how** we randomise
        matters enormously:

        - **Too small** → activations shrink layer by layer → **vanishing**
          signal, no learning.
        - **Too large** → activations blow up → **exploding** signal, saturated
          non-linearities.
        - The fix is to scale the random initial weights by the layer's
          **fan-in** (how many inputs it receives). A common choice for ReLU
          layers, called **He initialisation**, sets the standard deviation to

        $$\sigma = \sqrt{\frac{2}{n_{\text{in}}}}$$

        where $n_{\text{in}}$ is the number of inputs. (**Xavier**
        initialisation is a similar formula, $\sqrt{2/(n_{\text{in}}+n_{\text{out}})}$,
        often used with $\tanh$.)

        In plain words: when a neuron has **many inputs**, start its weights
        **small**, so the signal neither fades away nor blows up as it travels
        through the layers.

        scikit-learn does this for you; in PyTorch it is `nn.init.xavier_uniform_`
        / `nn.init.kaiming_normal_`.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 24</div>
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

    _fig, _ax = _plt.subplots(figsize=(7.5, 4.2))
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
    _ax.set_title("Weight scale controls signal propagation through the layers")
    _ax.legend()
    _ax.grid(alpha=0.3)
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.image(_buf, width="700px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">6 / 24</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Optimizers — beyond plain gradient descent

        Plain gradient descent (Session 3) is the starting point; the same
        update rule has three famous upgrades:

        - **SGD** — estimate the gradient on a **mini-batch** instead of the
          whole dataset: noisier, but far cheaper per step.
        - **Momentum** — remember the previous step and carry some of it forward,
          like a ball rolling downhill that picks up speed along a consistent
          direction: $v \leftarrow \beta v - \eta\,\nabla\mathcal{L}$, then
          $w \leftarrow w + v$.
        - **Adam** — the default in practice. It keeps a running average of the
          gradient and of how large the gradient has been, and uses them to
          scale each step. This adapts the learning rate **per weight**, so it is
          forgiving of a badly chosen $\eta$.

        In `MLPClassifier`: `solver="sgd"` (with `momentum=...`) or
        `solver="adam"`.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">7 / 24</div>
        """
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

    _fig, _ax = _plt.subplots(figsize=(7.5, 4.2))
    for _name, _kw, _c in _configs:
        _m = _MLP(
            hidden_layer_sizes=(32,), max_iter=2000, random_state=0, **_kw
        ).fit(_Xtr, _ytr)
        _ax.plot(_m.loss_curve_, color=_c,
                 label=f"{_name} (test {_m.score(_Xte, _yte):.2f})")

    _ax.set_xlabel("iteration")
    _ax.set_ylabel("loss")
    _ax.set_title("Optimizers compared on the two-moons task")
    _ax.legend()
    _ax.grid(alpha=0.3)
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.image(_buf, width="700px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">8 / 24</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Regularization — keeping the network honest

        A flexible network can **memorise** the training set. Regularization
        adds a cost for complexity, so the model prefers simpler solutions:

        - **Weight decay (L2)** — penalise large weights with
          `MLPClassifier(alpha=...)` (larger `alpha` = stronger). This is
          exactly the penalty used by `Ridge` regression.
        - **Dropout** — randomly switch off neurons during training so the
          network cannot rely on any single one (`MLPClassifier` has no
          dropout; modern frameworks such as PyTorch do).
        - **Early stopping** — stop when the **validation** loss stops
          improving: `MLPClassifier(early_stopping=True, validation_fraction=0.1)`.

        Regularization trades a little **training** accuracy for a lot of
        **test** accuracy.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">9 / 24</div>
        """
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

    _fig, _ax = _plt.subplots(figsize=(7, 4))
    _ax.semilogx(_alphas, _tr_acc, "o-", color="#dc2626", label="train")
    _ax.semilogx(_alphas, _te_acc, "o-", color="#16a34a", label="test")
    _ax.set_xlabel("alpha (L2 strength)")
    _ax.set_ylabel("accuracy")
    _ax.set_title("Weight decay: a little train accuracy buys a lot of test accuracy")
    _ax.legend()
    _ax.grid(alpha=0.3)
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.image(_buf, width="620px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">10 / 24</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Capacity — width and depth

        - **Width** — more units in a hidden layer.
        - **Depth** — more hidden layers, each building features on top of the
          previous one (edges → parts → objects, in vision).

        More capacity fits more complex boundaries — and overfits sooner.
        Watch the gap between **train and validation** accuracy, and reach for
        **early stopping** when it opens.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">11 / 24</div>
        """
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

    _X, _y = _make_moons(n_samples=500, noise=0.35, random_state=0)
    _Xtr, _Xte, _ytr, _yte = _split(
        _X, _y, test_size=0.3, random_state=0, stratify=_y
    )
    _sc = _Scaler().fit(_Xtr)
    _Xtr, _Xte = _sc.transform(_Xtr), _sc.transform(_Xte)

    _xx, _yy = _np.meshgrid(
        _np.linspace(_Xtr[:, 0].min() - 0.5, _Xtr[:, 0].max() + 0.5, 250),
        _np.linspace(_Xtr[:, 1].min() - 0.5, _Xtr[:, 1].max() + 0.5, 250),
    )
    _archs = [(32,), (32, 32), (32, 32, 32)]
    _fig, _axes = _plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for _ax, _arch in zip(_axes, _archs):
        _m = _MLP(
            hidden_layer_sizes=_arch, max_iter=2000, random_state=0
        ).fit(_Xtr, _ytr)
        _Z = _m.predict(_np.c_[_xx.ravel(), _yy.ravel()]).reshape(_xx.shape)
        _ax.contourf(_xx, _yy, _Z, alpha=0.25, cmap="RdYlGn")
        _ax.scatter(_Xtr[_ytr == 0, 0], _Xtr[_ytr == 0, 1], color="#dc2626", s=12)
        _ax.scatter(_Xtr[_ytr == 1, 0], _Xtr[_ytr == 1, 1], color="#16a34a", s=12)
        _ax.set_title(f"{len(_arch)} hidden layer(s) · test {_m.score(_Xte, _yte):.2f}")
        _ax.set_aspect("equal")
    _fig.tight_layout()
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.image(_buf, width="900px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">12 / 24</div>"""),
        ]
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
        ## Interactive demo — capacity and activation

        Bigger hidden layers can draw more complex boundaries — but also
        overfit more. Switch the activation function and see how training
        changes. (One hidden unit cannot even bend the boundary!)

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">13 / 24</div>
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
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">14 / 24</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Part 2 — Going beyond

        ## Multi-class: softmax output

        With $K$ classes, the last layer has $K$ neurons. **Softmax** turns their
        raw scores into a probability distribution:

        $$\hat{p}_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}, \qquad \sum_k \hat{p}_k = 1$$

        In plain words: exponentiate each score (to make them positive), then
        divide by the total — so the $K$ outputs become probabilities that add
        up to 1.

        The loss is the multi-class cross-entropy — sklearn handles all of this
        internally (`MLPClassifier` automatically uses softmax + cross-entropy
        for multi-class targets).

        Demo: 8×8 handwritten digits (1797 samples, 10 classes).

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">15 / 24</div>
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

    _mlp = _MLP(hidden_layer_sizes=(24,), max_iter=200, random_state=0).fit(_Xtr, _ytr)
    _acc = _mlp.score(_Xte, _yte)

    _fig, _axes = _plt.subplots(1, 2, figsize=(12, 4.4),
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
    mo.vstack(
        [
            mo.image(_buf, width="860px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">16 / 24</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Convolutional neural networks — an overview

        So far every neuron sees **all** inputs. That is wasteful for images:

        - **Locality** — nearby pixels matter together.
        - **Translation invariance** — a cat is a cat wherever it appears.

        A **convolution** slides a small kernel across the image, sharing the
        same weights everywhere. This cuts the number of parameters
        dramatically and bakes in the right **inductive bias** for images.
        **Pooling** then downsamples the feature maps.

        The training recipe is unchanged — still backpropagation + gradient
        descent. Below: a digit and an edge-detecting convolution.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">17 / 24</div>
        """
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np
    from scipy.signal import convolve2d as _conv
    from sklearn.datasets import load_digits as _load_digits

    _digits = _load_digits()
    _img = _digits.images[0]
    _kernel = _np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=float)
    _edges = _np.abs(_conv(_img, _kernel, mode="same"))

    _fig, _axes = _plt.subplots(1, 3, figsize=(10, 3.6))
    _axes[0].imshow(_img, cmap="gray_r")
    _axes[0].set_title("input digit")
    _axes[1].imshow(_kernel, cmap="RdBu", vmin=-1, vmax=1)
    _axes[1].set_title("3×3 kernel")
    _axes[2].imshow(_edges, cmap="gray_r")
    _axes[2].set_title("feature map (|edges|)")
    for _ax in _axes:
        _ax.set_xticks([])
        _ax.set_yticks([])
    _fig.tight_layout()
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.image(_buf, width="820px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">18 / 24</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Transformers — attention is all you need

        Convolutions assume **locality**; sequences (text, time series) often
        need **long-range** relationships. The **transformer** (Vaswani et al.,
        2017) replaces convolution with **self-attention**:

        - every token computes a **query**, a **key** and a **value**;
        - each token attends to **every other** token, weighted by
          query–key similarity;
        - **multi-head** attention lets it look at several relationships at once;
        - **positional encodings** inject order, since attention itself is
          permutation-invariant.

        Transformers are **not** a different kind of learning — still layers,
        activations, backpropagation and Adam. They are the architecture behind
        **BERT, GPT and modern LLMs**, and increasingly vision
        (ViT) and time series.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">19 / 24</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Autoencoders — learning without labels

        An **autoencoder** is a network trained to reconstruct its own input
        through a narrow **bottleneck**:

        $$x \;\rightarrow\; \text{encoder} \;\rightarrow\; z \;\rightarrow\;
        \text{decoder} \;\rightarrow\; \hat{x}$$

        Because the bottleneck is small, the network must learn a compact
        **representation** $z$ that keeps the important structure. No labels
        are needed — this is **representation learning**, a form of the
        unsupervised machine learning from Lecture 1, Session 1. Autoencoders are
        used for compression, denoising and pretraining.

        Below: a 64 → 32 → **8** → 32 → 64 autoencoder on handwritten digits,
        trained with `MLPRegressor` on its own input.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">20 / 24</div>
        """
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.datasets import load_digits as _load_digits
    from sklearn.neural_network import MLPRegressor as _MLPR

    _digits = _load_digits()
    _X = _digits.data / 16.0

    _ae = _MLPR(hidden_layer_sizes=(32, 8, 32), max_iter=400, random_state=0)
    _ae.fit(_X, _X)
    _Xhat = _np.clip(_ae.predict(_X[:8]), 0, 1)

    _fig, _axes = _plt.subplots(2, 8, figsize=(12, 3.4))
    for _i in range(8):
        _axes[0, _i].imshow(_digits.images[_i], cmap="gray_r")
        _axes[1, _i].imshow(_Xhat[_i].reshape(8, 8), cmap="gray_r")
        _axes[0, _i].set_xticks([])
        _axes[0, _i].set_yticks([])
        _axes[1, _i].set_xticks([])
        _axes[1, _i].set_yticks([])
    _axes[0, 0].set_ylabel("original", fontsize=9)
    _axes[1, 0].set_ylabel("reconstructed", fontsize=9)
    _fig.tight_layout()
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.image(_buf, width="900px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">21 / 24</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The three model families of this course

        | | logistic regression | trees / forests | neural networks |
        |---|---|---|---|
        | boundary | straight line | axis-aligned steps | smooth, arbitrary |
        | interpretability | high (weights) | tree = flowchart | low |
        | feature scaling | recommended | not needed | required |
        | training cost | seconds | seconds–minutes | minutes–hours |
        | best at | fast baselines, calibrated probabilities | tabular data | images, text, signals |

        Rule of thumb: start with logistic regression, try a random forest,
        reach for a neural network when the data has *structure* (pixels,
        sequences) or the patterns are truly smooth.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">22 / 24</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Summary

        - **Components** are the practical levers of a network: the
          **activation** (sigmoid/tanh/ReLU/softmax), **weight initialization**
          (Xavier/He), the **optimizer** (SGD → momentum → Adam),
          **regularization** (weight decay, dropout, early stopping) and
          **capacity** (width & depth).
        - **Softmax** extends networks to many classes; **CNNs** add the right
          inductive bias for images.
        - **Transformers** replace convolution with **self-attention** — the
          architecture behind modern LLMs; training is still backpropagation.
        - **Autoencoders** learn representations **without labels**.
        - Rule of thumb: **logistic regression → random forest → neural
          network**, depending on the data.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">23 / 24</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Where to go next

        - **Exercise:** `notebooks/02/04_components_and_beyond_{beginner,intermediate,advanced}.ipynb`
          — explore the digits data, compare activations/optimizers with
          scikit-learn, and implement **SGD, momentum and Adam from scratch**.
        - This wraps the lecture series — the final project is an end-to-end
          ML pipeline on a dataset of your choice. Happy learning!

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">24 / 24</div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
