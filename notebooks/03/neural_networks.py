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
# Lecture 3 — Neural Networks.
# Run locally with `marimo edit notebooks/03/neural_networks.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).

import marimo

__generated_with = "0.17.6"
app = marimo.App(
    width="medium",
    layout_file="layouts/neural_networks.slides.json",
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
        # Neural Networks

        **Machine Learning with Python** — Lecture 3

        EUGLOH — *Problem Solving Using Open-Source Languages; R and Python*

        University of Novi Sad
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Today's lecture

        - Where linear models (and trees!) hit a wall: smooth non-linear boundaries
        - **Neurons** and **activation functions**
        - **Multi-layer perceptrons**: forward pass, backpropagation, gradient descent
        - A neural network **written from scratch in numpy** — then the same in sklearn
        - Practical tips: scaling, capacity, early stopping, multi-class softmax
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Part 1 — Why we need non-linear models

        Concentric circles: **no straight line** can separate the classes.
        A tree can only cut with axis-aligned rules — it would need many
        rectangles to imitate a circle.

        Let's watch logistic regression fail.
        """
    )
    return


@app.cell
def _(mo):
    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.datasets import make_circles as _make_circles
    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.preprocessing import StandardScaler as _Scaler

    _X, _y = _make_circles(n_samples=400, factor=0.4, noise=0.1, random_state=0)
    nn_X, nn_y = _X, _y
    nn_Xs = _Scaler().fit_transform(_X)  # scaled copy — used for training

    _clf = _LR().fit(nn_Xs, nn_y)
    _acc = _clf.score(nn_Xs, nn_y)

    _xx, _yy = _np.meshgrid(
        _np.linspace(nn_Xs[:, 0].min() - 0.5, nn_Xs[:, 0].max() + 0.5, 250),
        _np.linspace(nn_Xs[:, 1].min() - 0.5, nn_Xs[:, 1].max() + 0.5, 250),
    )
    _Z = _clf.predict(_np.c_[_xx.ravel(), _yy.ravel()]).reshape(_xx.shape)

    _fig, _ax = _plt.subplots(figsize=(5.5, 5))
    _ax.contourf(_xx, _yy, _Z, alpha=0.25, cmap="RdYlGn")
    _ax.scatter(nn_Xs[nn_y == 0, 0], nn_Xs[nn_y == 0, 1], color="#dc2626", s=16, label="class 0")
    _ax.scatter(nn_Xs[nn_y == 1, 0], nn_Xs[nn_y == 1, 1], color="#16a34a", s=16, label="class 1")
    _ax.set_title(f"Logistic regression — accuracy {_acc:.2f}. Stuck at a line.")
    _ax.set_aspect("equal")
    _ax.legend()
    mo.as_html(_fig)
    _plt.close(_fig)
    return nn_X, nn_Xs, nn_y


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The artificial neuron

        A neuron computes a weighted sum, then applies a non-linear
        **activation**:

        $$z = w^\top x + b, \qquad a = \phi(z)$$

        Recognise it? **Logistic regression is exactly one neuron** with the
        sigmoid activation. A neural network is just *many neurons stacked in
        layers*, each feeding the next.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Activation functions

        | function | formula | range | notes |
        |---|---|---|---|
        | sigmoid | $\sigma(z) = 1/(1+e^{-z})$ | $(0,1)$ | probability-like; saturates → vanishing gradients |
        | tanh | $\tanh(z)$ | $(-1,1)$ | zero-centred; hidden-layer classic |
        | ReLU | $\max(0, z)$ | $[0,\infty)$ | cheap, no saturation for $z>0$ — today's default |

        The non-linearity is the whole point: compose enough of them and the
        network can approximate **any** smooth function.
        """
    )
    return


@app.cell
def _(mo):
    import matplotlib.pyplot as _plt
    import numpy as _np

    _z = _np.linspace(-4, 4, 300)
    _fig, _ax = _plt.subplots(figsize=(7.5, 3.8))
    _ax.plot(_z, 1 / (1 + _np.exp(-_z)), lw=2, label="sigmoid", color="#2563eb")
    _ax.plot(_z, _np.tanh(_z), lw=2, label="tanh", color="#16a34a")
    _ax.plot(_z, _np.maximum(0, _z), lw=2, label="ReLU", color="#dc2626")
    _ax.axhline(0, color="gray", lw=0.5)
    _ax.axvline(0, color="gray", lw=0.5)
    _ax.set_title("Activation functions")
    _ax.legend()
    _ax.grid(alpha=0.3)
    mo.as_html(_fig)
    _plt.close(_fig)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The multi-layer perceptron (MLP)

        Layers of neurons: **input → hidden → output**. For one hidden layer
        with $H$ units (shapes for $n$ samples, $d$ features):

        $$Z_1 = XW_1 + b_1 \quad (n \times H), \qquad A_1 = \tanh(Z_1)$$

        $$Z_2 = A_1 W_2 + b_2 \quad (n \times 1), \qquad \hat{p} = \sigma(Z_2)$$

        Everything is **matrix multiplication** — the reason GPUs are so good
        at this.

        **Universal approximation:** one hidden layer with enough units can
        approximate any continuous function on a compact set. The catch:
        nobody tells you *how many* units — you must learn them from data.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Training: loss + backpropagation

        Same recipe as logistic regression — cross-entropy loss, gradient
        descent. The only new skill is computing the gradient through the
        layers, which is the **chain rule** applied layer by layer
        (*backpropagation*). For our 1-hidden-layer network:

        $$\delta_2 = \frac{\hat{p} - y}{n} \quad \text{(output error)}$$

        $$\frac{\partial \mathcal{L}}{\partial W_2} = A_1^\top \delta_2, \qquad \frac{\partial \mathcal{L}}{\partial b_2} = \textstyle\sum_i \delta_{2,i}$$

        $$\delta_1 = (\delta_2 W_2^\top) \odot (1 - A_1^2) \quad \text{(chain rule through tanh)}$$

        $$\frac{\partial \mathcal{L}}{\partial W_1} = X^\top \delta_1, \qquad \frac{\partial \mathcal{L}}{\partial b_1} = \textstyle\sum_i \delta_{1,i}$$

        You will implement exactly these four lines in the exercise.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## An MLP by hand, in numpy

        2 inputs → 16 hidden units (tanh) → 1 output (sigmoid), trained with
        full-batch gradient descent on the circles dataset.

        TODO: walk through the forward pass, then the four gradient lines.
        """
    )
    return


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

    _lr, _iters = 0.5, 4000
    nn_losses = []
    for _i in range(_iters):
        _Z1 = nn_Xs @ W1h + b1h
        _A1 = _np.tanh(_Z1)
        _Z2 = _A1 @ W2h + b2h
        _P = 1.0 / (1.0 + _np.exp(-_np.clip(_Z2, -30, 30)))

        _delta2 = (_P - nn_y.reshape(-1, 1)) / _n
        _dW2 = _A1.T @ _delta2
        _db2 = _delta2.sum(axis=0)
        _delta1 = (_delta2 @ W2h.T) * (1 - _A1**2)
        _dW1 = nn_Xs.T @ _delta1
        _db1 = _delta1.sum(axis=0)

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
    print(f"hand-written MLP (2 → 16 → 1): accuracy = {nn_acc:.3f}")
    return W1h, W2h, b1h, b2h, nn_acc, nn_losses


@app.cell
def _(b1h, b2h, mo, nn_X, nn_Xs, nn_acc, nn_losses, nn_y, W1h, W2h):
    import matplotlib.pyplot as _plt
    import numpy as _np

    _xx, _yy = _np.meshgrid(
        _np.linspace(nn_Xs[:, 0].min() - 0.5, nn_Xs[:, 0].max() + 0.5, 250),
        _np.linspace(nn_Xs[:, 1].min() - 0.5, nn_Xs[:, 1].max() + 0.5, 250),
    )
    _grid = _np.c_[_xx.ravel(), _yy.ravel()]
    _P = 1.0 / (1.0 + _np.exp(-(_np.tanh(_grid @ W1h + b1h) @ W2h + b2h)))
    _Z = _P.reshape(_xx.shape)

    _fig, _axes = _plt.subplots(1, 2, figsize=(11, 4.4))
    _axes[0].plot(_np.arange(len(nn_losses)) * 100, nn_losses, color="#2563eb")
    _axes[0].set_xlabel("epoch")
    _axes[0].set_title("Cross-entropy during training")
    _axes[1].contourf(_xx, _yy, _Z, levels=20, cmap="RdYlGn", alpha=0.7)
    _axes[1].scatter(nn_X[nn_y == 0, 0], nn_X[nn_y == 0, 1], color="#dc2626", s=14)
    _axes[1].scatter(nn_X[nn_y == 1, 0], nn_X[nn_y == 1, 1], color="#16a34a", s=14)
    _axes[1].set_title(f"Learned boundary (by hand, acc {nn_acc:.2f})")
    _axes[1].set_aspect("equal")
    mo.as_html(_fig)
    _plt.close(_fig)
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

        `MLPClassifier` uses Adam (a smarter gradient-descent variant), tracks
        its loss curve, and supports early stopping.
        """
    )
    return


@app.cell
def _(mo, nn_Xs, nn_y):
    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.neural_network import MLPClassifier as _MLP

    _mlp = _MLP(
        hidden_layer_sizes=(16,), activation="tanh",
        solver="adam", max_iter=2000, random_state=0,
    ).fit(nn_Xs, nn_y)

    _xx, _yy = _np.meshgrid(
        _np.linspace(nn_Xs[:, 0].min() - 0.5, nn_Xs[:, 0].max() + 0.5, 250),
        _np.linspace(nn_Xs[:, 1].min() - 0.5, nn_Xs[:, 1].max() + 0.5, 250),
    )
    _Z = _mlp.predict(_np.c_[_xx.ravel(), _yy.ravel()]).reshape(_xx.shape)

    _fig, _axes = _plt.subplots(1, 2, figsize=(11, 4.4))
    _axes[0].contourf(_xx, _yy, _Z, levels=20, cmap="RdYlGn", alpha=0.7)
    _axes[0].scatter(nn_Xs[nn_y == 0, 0], nn_Xs[nn_y == 0, 1], color="#dc2626", s=14)
    _axes[0].scatter(nn_Xs[nn_y == 1, 0], nn_Xs[nn_y == 1, 1], color="#16a34a", s=14)
    _axes[0].set_title(f"MLPClassifier — accuracy {_mlp.score(nn_Xs, nn_y):.2f}")
    _axes[0].set_aspect("equal")
    _axes[1].plot(_mlp.loss_curve_, color="#dc2626")
    _axes[1].set_xlabel("iteration")
    _axes[1].set_title("sklearn's loss curve (Adam)")
    mo.as_html(_fig)
    _plt.close(_fig)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Interactive demo — capacity and activation

        Bigger hidden layers can draw more complex boundaries — but also
        overfit more. Switch the activation function and see how training
        changes.
        """
    )
    return


@app.cell
def _(mo):
    hidden_slider = mo.ui.slider(
        start=1, stop=64, step=1, value=8,
        label="hidden units", show_value=True,
    )
    act_dropdown = mo.ui.dropdown(
        options=["tanh", "relu", "logistic"], value="tanh",
        label="activation",
    )
    mo.hstack([hidden_slider, act_dropdown])
    return act_dropdown, hidden_slider


@app.cell
def _(act_dropdown, hidden_slider, mo, nn_X, nn_Xs, nn_y):
    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.neural_network import MLPClassifier as _MLP

    _mlp = _MLP(
        hidden_layer_sizes=(hidden_slider.value,),
        activation=act_dropdown.value,
        max_iter=3000,
        random_state=0,
    ).fit(nn_Xs, nn_y)

    _xx, _yy = _np.meshgrid(
        _np.linspace(nn_Xs[:, 0].min() - 0.5, nn_Xs[:, 0].max() + 0.5, 250),
        _np.linspace(nn_Xs[:, 1].min() - 0.5, nn_Xs[:, 1].max() + 0.5, 250),
    )
    _Z = _mlp.predict(_np.c_[_xx.ravel(), _yy.ravel()]).reshape(_xx.shape)

    _fig, _ax = _plt.subplots(figsize=(6, 5))
    _ax.contourf(_xx, _yy, _Z, alpha=0.25, cmap="RdYlGn")
    _ax.scatter(nn_X[nn_y == 0, 0], nn_X[nn_y == 0, 1], color="#dc2626", s=14)
    _ax.scatter(nn_X[nn_y == 1, 0], nn_X[nn_y == 1, 1], color="#16a34a", s=14)
    _ax.set_title(
        f"({hidden_slider.value},) {act_dropdown.value} · "
        f"accuracy {_mlp.score(nn_Xs, nn_y):.2f}"
    )
    _ax.set_aspect("equal")
    mo.as_html(_fig)
    _plt.close(_fig)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Practical considerations

        - **Scale your features** — MLPs hate features of wildly different ranges
        - **Weight init** — random but *not too large* (Xavier/He scaling; sklearn does this for you)
        - **Learning rate** — too small: slow; too large: divergence (bonus exercise!)
        - **Capacity** — more hidden units = more expressive, but watch the gap
          between train and validation accuracy
        - **Early stopping** — stop when validation loss stops improving
          (`MLPClassifier(early_stopping=True)`)

        The figure below: train vs test accuracy as the network grows — the
        classic overfitting gap.
        """
    )
    return


@app.cell
def _(mo):
    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.datasets import make_moons as _make_moons
    from sklearn.model_selection import train_test_split as _split
    from sklearn.neural_network import MLPClassifier as _MLP

    _X, _y = _make_moons(n_samples=400, noise=0.35, random_state=0)
    _Xtr, _Xte, _ytr, _yte = _split(_X, _y, test_size=0.3, random_state=0, stratify=_y)

    _sizes = [1, 2, 4, 8, 16, 32]
    _train_acc, _test_acc = [], []
    for _h in _sizes:
        _m = _MLP(hidden_layer_sizes=(_h,), max_iter=2000, random_state=0).fit(_Xtr, _ytr)
        _train_acc.append(_m.score(_Xtr, _ytr))
        _test_acc.append(_m.score(_Xte, _yte))

    _fig, _ax = _plt.subplots(figsize=(7, 4))
    _ax.plot(_sizes, _train_acc, "o-", color="#2563eb", label="train")
    _ax.plot(_sizes, _test_acc, "s-", color="#dc2626", label="test")
    _ax.set_xlabel("hidden units")
    _ax.set_ylabel("accuracy")
    _ax.set_xscale("log", base=2)
    _ax.set_xticks(_sizes, labels=[str(_s) for _s in _sizes])
    _ax.set_title("Bigger networks memorise the train set — but stop helping on test")
    _ax.legend()
    _ax.grid(alpha=0.3)
    mo.as_html(_fig)
    _plt.close(_fig)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Multi-class: softmax output

        With $K$ classes, the last layer has $K$ neurons and the **softmax**
        turns them into a probability distribution:

        $$\hat{p}_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}, \qquad \sum_k \hat{p}_k = 1$$

        The loss is the multi-class cross-entropy — sklearn handles all of
        this internally (`MLPClassifier` automatically uses softmax +
        cross-entropy for multi-class targets).

        Demo: 8×8 handwritten digits (1797 samples, 10 classes).
        """
    )
    return


@app.cell
def _(mo):
    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.datasets import load_digits as _load_digits
    from sklearn.metrics import ConfusionMatrixDisplay as _CMD
    from sklearn.model_selection import train_test_split as _split
    from sklearn.neural_network import MLPClassifier as _MLP

    _digits = _load_digits()
    _X = _digits.data / 16.0
    _y = _digits.target
    _Xtr, _Xte, _ytr, _yte = _split(_X, _y, test_size=0.3, random_state=0, stratify=_y)

    _mlp = _MLP(hidden_layer_sizes=(32, 32), max_iter=300, random_state=0).fit(_Xtr, _ytr)
    _acc = _mlp.score(_Xte, _yte)

    _fig, _axes = _plt.subplots(1, 2, figsize=(12, 4.6),
                                gridspec_kw={"width_ratios": [1, 1.4]})
    _axes[0].imshow(_digits.images[0], cmap="gray_r")
    _axes[0].set_title(f"an 8×8 digit — test accuracy {_acc:.3f}")
    _axes[0].set_xticks([])
    _axes[0].set_yticks([])
    _CMD.from_predictions(_yte, _mlp.predict(_Xte), ax=_axes[1], colorbar=False)
    _axes[1].set_title("Confusion matrix — which digits get confused?")
    _fig.tight_layout()
    mo.as_html(_fig)
    _plt.close(_fig)
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
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Summary

        - A neuron = linear model + activation; an MLP stacks them into layers
        - Training = cross-entropy + **backpropagation** (chain rule through
          the layers) + gradient descent — you built one **from scratch** and
          it drew a round boundary in a place no linear model could
        - Capacity, scaling, learning rate and early stopping are the practical
          levers — and the usual suspects behind overfitting
        - Softmax extends everything to many classes
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Where to go next

        - **Exercise:** `notebooks/03/neural_networks_exercise.ipynb`
          — implement forward + backward pass by hand, compare against
          `MLPClassifier`, explore capacity on moons and digits.
        - This wraps the lecture series — the final project is an end-to-end
          ML pipeline on a dataset of your choice. Happy learning!
        """
    )
    return


if __name__ == "__main__":
    app.run()
