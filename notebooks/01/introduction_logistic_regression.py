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
# Lecture 1 — Introduction to Machine Learning & Logistic Regression.
# Run locally with `marimo edit notebooks/01/introduction_logistic_regression.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).

import marimo

__generated_with = "0.17.6"
app = marimo.App(
    width="medium",
    layout_file="layouts/introduction_logistic_regression.slides.json",
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
        # Introduction to Machine Learning & Logistic Regression

        **Machine Learning with Python** — Lecture 1

        EUGLOH — *Problem Solving Using Open-Source Languages; R and Python*

        University of Novi Sad
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
        - Today's exercise: `notebooks/01/introduction_logistic_regression_exercise.ipynb`
        - All exercises use `numpy`, `matplotlib` and `scikit-learn` — nothing else to install

        Questions are very welcome — ask early, ask often.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Part 1 — What is machine learning?

        > *"A computer program is said to learn from experience E with respect to
        > some class of tasks T and performance measure P, if its performance at
        > tasks in T, as measured by P, improves with experience E."*

        — Mitchell, 1997
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        In one sentence:

        **Machine learning is the discipline of building systems that learn rules
        from data, instead of being explicitly programmed.**

        - Traditional programming: *data + program → output*
        - Machine learning: *data + output → program*

        The learned **model** is then applied to new, unseen data.
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
        - Problems where **data is abundant** but theory is incomplete

        ML is not magic — it is a tool that shines when the data is right and
        the question is well-posed.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Part 2 — Types of machine learning

        | Type | What the algorithm sees | Goal |
        |---|---|---|
        | **Supervised** | inputs **+ labels** | predict the label of new inputs |
        | **Unsupervised** | inputs only | discover structure (clusters, low-dimensional representations) |
        | **Semi-supervised** | few labels + many unlabeled inputs | leverage both |
        | **Self-supervised** | inputs only | create labels from the data itself (denoising, masked words) |
        | **Reinforcement** | interactions + rewards | learn actions that maximise reward |

        This course is mostly about **supervised learning**.
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

        Algorithms in this course: **logistic regression** (today),
        **decision trees & random forests** (lecture 2), **neural networks** (lecture 3).
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
          often expensive — think medical images). Key idea: unlabeled points
          reveal the *shape* of the data, which constrains the decision boundary.
        - **Self-supervised:** the data creates its own supervision — predict a
          masked word, reconstruct a denoised image. The engine behind modern LLMs.
        - **Reinforcement learning:** an agent acts, the environment rewards.
          Out of scope for this course.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## One dataset, three paradigms

        The slider controls **how many points are labeled**:

        - 100 % labeled → **supervised** classification
        - a few labeled → **semi-supervised** learning
        - none labeled → **unsupervised** learning (e.g. clustering)
        """
    )
    return


@app.cell
def _(mo):
    label_fraction = mo.ui.slider(
        start=0.0, stop=1.0, step=0.05, value=0.1,
        label="Fraction of labeled points", show_value=True,
    )
    label_fraction
    return (label_fraction,)


@app.cell
def _(label_fraction, mo):
    import numpy as _np
    import matplotlib.pyplot as _plt
    from sklearn.datasets import make_moons as _make_moons

    _X, _y = _make_moons(n_samples=150, noise=0.25, random_state=0)
    _n = len(_y)
    _k = int(round(label_fraction.value * _n))
    _order = _np.random.default_rng(0).permutation(_n)
    _labeled = _order[:_k]

    _fig, _ax = _plt.subplots(figsize=(6.5, 4.5))
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
    mo.as_html(_fig)
    _plt.close(_fig)
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

        The demo below fits logistic regression on the two-moons dataset with
        polynomial features of increasing degree — watch the boundary wiggle.
        """
    )
    return


@app.cell
def _(mo):
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
    for _ax, _deg in zip(_axes, [1, 4, 15]):
        _pipe = _mkpipe(_Poly(degree=_deg), _Scaler(), _LR(max_iter=5000))
        _pipe.fit(_X, _y)
        _acc = _pipe.score(_X, _y)
        _zz = _pipe.predict(_np.c_[_xx.ravel(), _yy.ravel()]).reshape(_xx.shape)
        _ax.contourf(_xx, _yy, _zz, alpha=0.25, cmap="RdYlGn")
        _ax.scatter(_X[_y == 0, 0], _X[_y == 0, 1], color="#dc2626", s=14)
        _ax.scatter(_X[_y == 1, 0], _X[_y == 1, 1], color="#16a34a", s=14)
        _ax.set_title(f"polynomial degree {_deg} · train acc {_pipe.score(_X, _y):.2f}")
        _ax.set_aspect("equal")
    mo.as_html(_fig)
    _plt.close(_fig)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The datasets of this course

        Everything runs offline — the datasets ship with scikit-learn.
        """
    )
    return


@app.cell
def _(mo):
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

    _fig, _axes = _plt.subplots(2, 3, figsize=(11, 6.4))

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
    mo.as_html(_fig)
    _plt.close(_fig)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Part 3 — Logistic regression

        ## Recap: the linear model

        Linear regression predicts a **continuous** number with a linear function:

        $$\hat{y} = w^\top x + b$$

        But what if we want to predict a **class label** (spam / not spam)?
        We could threshold a linear model — ugly and unbounded. Instead:
        squash the linear output through the **sigmoid function**.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The sigmoid function

        $$\sigma(z) = \frac{1}{1 + e^{-z}}$$

        - maps any real number to $(0, 1)$ — a probability
        - smooth, differentiable everywhere (we will need the derivative)
        """
    )
    return


@app.cell
def _(mo):
    import matplotlib.pyplot as _plt
    import numpy as _np

    _z = _np.linspace(-6, 6, 300)
    _sig = 1.0 / (1.0 + _np.exp(-_z))

    _fig, _ax = _plt.subplots(figsize=(7, 3.5))
    _ax.plot(_z, _sig, color="#2563eb", linewidth=2.5)
    _ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    _ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    _ax.set_xlabel("z")
    _ax.set_ylabel("σ(z)")
    _ax.set_title("The sigmoid function")
    _ax.grid(alpha=0.3)
    mo.as_html(_fig)
    _plt.close(_fig)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The logistic regression model

        $$\hat{p}(y{=}1 \mid x) = \sigma(w^\top x + b) = \frac{1}{1 + e^{-(w^\top x + b)}}$$

        - The output is a **probability** in $[0, 1]$
        - Predict class 1 if $\hat{p} \geq 0.5$ (the *decision threshold*)
        - The boundary $w^\top x + b = 0$ is a **hyperplane** — logistic
          regression is still a *linear* classifier
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The loss: cross-entropy

        Mean squared error + sigmoid gives a **non-convex** surface. Instead we
        maximise the likelihood of the labels (a Bernoulli model), which is
        equivalent to minimising the **binary cross-entropy**:

        $$\mathcal{L}(w, b) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log \hat{p}_i + (1 - y_i) \log (1 - \hat{p}_i) \right]$$

        - Convex in $w$ → gradient descent finds the **global** minimum
        - Gradients (derived with the chain rule):

        $$\nabla_w \mathcal{L} = \frac{1}{n} X^\top (\hat{p} - y), \qquad \nabla_b \mathcal{L} = \frac{1}{n} \sum_i (\hat{p}_i - y_i)$$
        """
    )
    return


@app.cell
def _():
    import numpy as _np
    from sklearn.datasets import make_blobs as _make_blobs

    blobs_X, blobs_y = _make_blobs(
        n_samples=200, centers=[(-2.0, -2.0), (2.0, 2.0)],
        cluster_std=1.2, random_state=0,
    )
    return blobs_X, blobs_y


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Logistic regression by hand (numpy)

        ~15 lines: predict, measure the loss, nudge the weights downhill.

        TODO: walk through this line by line in the lecture.
        """
    )
    return


@app.cell
def _(blobs_X, blobs_y, mo):
    import matplotlib.pyplot as _plt
    import numpy as _np

    def _sigmoid(z):
        return 1.0 / (1.0 + _np.exp(-z))

    def fit_logistic_gd(X, y, lr=0.5, n_iter=500):
        """Train binary logistic regression with full-batch gradient descent."""
        n, d = X.shape
        w = _np.zeros(d)
        b = 0.0
        losses = []
        for _ in range(n_iter):
            p = _sigmoid(X @ w + b)
            losses.append(
                -_np.mean(y * _np.log(p + 1e-12) + (1 - y) * _np.log(1 - p + 1e-12))
            )
            w -= lr * (X.T @ (p - y)) / n
            b -= lr * _np.mean(p - y)
        return w, b, losses

    w_hand, b_hand, losses_hand = fit_logistic_gd(blobs_X, blobs_y)
    acc_hand = _np.mean((_sigmoid(blobs_X @ w_hand + b_hand) >= 0.5) == blobs_y)
    print(f"by hand: w = {w_hand.round(3)}, b = {b_hand:+.3f}, accuracy = {acc_hand:.3f}")

    _xx, _yy = _np.meshgrid(
        _np.linspace(blobs_X[:, 0].min() - 1, blobs_X[:, 0].max() + 1, 250),
        _np.linspace(blobs_X[:, 1].min() - 1, blobs_X[:, 1].max() + 1, 250),
    )
    _pp = _sigmoid(
        _np.c_[_xx.ravel(), _yy.ravel()] @ w_hand + b_hand
    ).reshape(_xx.shape)

    _fig, _axes = _plt.subplots(1, 2, figsize=(11, 4.2))
    _axes[0].plot(losses_hand, color="#2563eb")
    _axes[0].set_xlabel("epoch")
    _axes[0].set_title("Cross-entropy during training")
    _cs = _axes[1].contourf(_xx, _yy, _pp, levels=20, cmap="RdYlGn", alpha=0.7)
    _axes[1].scatter(blobs_X[blobs_y == 0, 0], blobs_X[blobs_y == 0, 1],
                     color="#dc2626", s=16, label="class 0")
    _axes[1].scatter(blobs_X[blobs_y == 1, 0], blobs_X[blobs_y == 1, 1],
                     color="#16a34a", s=16, label="class 1")
    _axes[1].set_title(f"Decision boundary (by hand, acc {acc_hand:.2f})")
    _axes[1].legend()
    _fig.colorbar(_cs, ax=_axes[1], label="p(class 1)")
    mo.as_html(_fig)
    _plt.close(_fig)
    return acc_hand, b_hand, fit_logistic_gd, w_hand


@app.cell
def _(acc_hand, b_hand, blobs_X, blobs_y, mo, w_hand):
    from sklearn.linear_model import LogisticRegression as _LR

    logreg_clf = _LR().fit(blobs_X, blobs_y)
    print(f"scikit-learn: w = {logreg_clf.coef_[0].round(3)}, "
          f"b = {logreg_clf.intercept_[0]:+.3f}, "
          f"accuracy = {logreg_clf.score(blobs_X, blobs_y):.3f}")
    print(f"by hand:      w = {w_hand.round(3)}, b = {b_hand:+.3f}, accuracy = {acc_hand:.3f}")
    print("(sklearn's weights are slightly smaller — it applies L2 regularisation by default)")
    return (logreg_clf,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Practical considerations

        - **Feature scaling** — standardise features (mean 0, std 1) so gradient
          descent converges quickly and features are treated fairly.
        - **Regularisation** — sklearn penalises large weights by default
          (`C = 1/λ`, larger `C` → weaker regularisation).
        - **Probabilities vs labels** — `predict_proba` gives calibrated
          probabilities; `predict` applies the 0.5 threshold.
        - **Multi-class** — sklearn trains one-vs-rest by default (next slides).
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Interactive demo — class separation

        The slider controls how far apart the two classes are. Watch the
        decision boundary and the accuracy react.
        """
    )
    return


@app.cell
def _(mo):
    sep_slider = mo.ui.slider(
        start=0.5, stop=4.0, step=0.1, value=2.0,
        label="Class separation", show_value=True,
    )
    sep_slider
    return (sep_slider,)


@app.cell
def _(mo, sep_slider):
    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.metrics import accuracy_score as _acc

    _rng = _np.random.default_rng(42)
    _sep = sep_slider.value
    _Xp = _rng.normal(loc=[_sep, _sep], scale=[1.0, 1.0], size=(80, 2))
    _Xn = _rng.normal(loc=[-_sep, -_sep], scale=[1.0, 1.0], size=(80, 2))
    _Xd = _np.vstack([_Xp, _Xn])
    _yd = _np.concatenate([_np.ones(80), _np.zeros(80)])
    _clf = _LR().fit(_Xd, _yd)
    _train_acc = _acc(_yd, _clf.predict(_Xd))

    _x0, _x1 = _Xd[:, 0].min() - 1, _Xd[:, 0].max() + 1
    _y0, _y1 = _Xd[:, 1].min() - 1, _Xd[:, 1].max() + 1
    _xx, _yy = _np.meshgrid(_np.linspace(_x0, _x1, 200), _np.linspace(_y0, _y1, 200))
    _Z = _clf.predict(_np.c_[_xx.ravel(), _yy.ravel()]).reshape(_xx.shape)

    _fig, _ax = _plt.subplots(figsize=(6, 5))
    _ax.contourf(_xx, _yy, _Z, alpha=0.2, cmap="RdYlGn")
    _ax.scatter(_Xd[_yd == 1, 0], _Xd[_yd == 1, 1], color="#16a34a", alpha=0.7, label="class 1")
    _ax.scatter(_Xd[_yd == 0, 0], _Xd[_yd == 0, 1], color="#dc2626", alpha=0.7, label="class 0")
    _ax.set_xlabel("$x_1$")
    _ax.set_ylabel("$x_2$")
    _ax.set_title(f"Training accuracy: {_train_acc:.3f}")
    _ax.legend()
    _ax.set_aspect("equal")
    mo.as_html(_fig)
    _plt.close(_fig)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Interactive demo — the decision threshold

        A classifier is more than "class A or B": it outputs a **probability**,
        and *we* choose where to cut. Move the threshold and watch which errors
        you trade: false positives (amber) vs false negatives (red).
        """
    )
    return


@app.cell
def _(mo):
    thr_slider = mo.ui.slider(
        start=0.05, stop=0.95, step=0.05, value=0.5,
        label="Decision threshold", show_value=True,
    )
    thr_slider
    return (thr_slider,)


@app.cell
def _(blobs_X, blobs_y, logreg_clf, mo, thr_slider):
    import matplotlib.pyplot as _plt
    import numpy as _np

    _p = logreg_clf.predict_proba(blobs_X)[:, 1]
    _t = thr_slider.value
    _pred = (_p >= _t).astype(int)

    _tp = int(((_pred == 1) & (blobs_y == 1)).sum())
    _tn = int(((_pred == 0) & (blobs_y == 0)).sum())
    _fp = int(((_pred == 1) & (blobs_y == 0)).sum())
    _fn = int(((_pred == 0) & (blobs_y == 1)).sum())

    _fig, _ax = _plt.subplots(figsize=(6.5, 5))
    for _mask, _color, _label in [
        ((_pred == 1) & (blobs_y == 1), "#16a34a", f"true positives ({_tp})"),
        ((_pred == 0) & (blobs_y == 0), "#94a3b8", f"true negatives ({_tn})"),
        ((_pred == 1) & (blobs_y == 0), "#f59e0b", f"false positives ({_fp})"),
        ((_pred == 0) & (blobs_y == 1), "#dc2626", f"false negatives ({_fn})"),
    ]:
        _ax.scatter(blobs_X[_mask, 0], blobs_X[_mask, 1], color=_color, s=28, label=_label)
    _ax.set_title(f"threshold = {_t:.2f} — which error do you prefer?")
    _ax.legend(loc="upper left", fontsize=8)
    _ax.set_aspect("equal")
    mo.as_html(_fig)
    _plt.close(_fig)
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
        """
    )
    return


@app.cell
def _(mo):
    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.datasets import load_iris as _load_iris
    from sklearn.linear_model import LogisticRegression as _LR

    _iris = _load_iris()
    _X = _iris.data[:, 2:4]  # petal length & width
    _y = _iris.target
    _clf = _LR(max_iter=1000).fit(_X, _y)

    _x0, _x1 = _X[:, 0].min() - 0.5, _X[:, 0].max() + 0.5
    _y0, _y1 = _X[:, 1].min() - 0.5, _X[:, 1].max() + 0.5
    _xx, _yy = _np.meshgrid(_np.linspace(_x0, _x1, 300), _np.linspace(_y0, _y1, 300))
    _Z = _clf.predict(_np.c_[_xx.ravel(), _yy.ravel()]).reshape(_xx.shape)

    _fig, _ax = _plt.subplots(figsize=(6.5, 5))
    _ax.contourf(_xx, _yy, _Z, alpha=0.25, cmap="RdYlGn")
    for _k, _name in enumerate(_iris.target_names):
        _ax.scatter(_X[_y == _k, 0], _X[_y == _k, 1], s=22, label=_name)
    _ax.set_xlabel("petal length (cm)")
    _ax.set_ylabel("petal width (cm)")
    _ax.set_title(f"Logistic regression on iris — accuracy {_clf.score(_X, _y):.2f}")
    _ax.legend()
    mo.as_html(_fig)
    _plt.close(_fig)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Evaluating a classifier (a taste)

        - **Accuracy** — fraction of correct predictions. Fine when classes are balanced.
        - **Confusion matrix** — TP / FP / FN / TN per class; shows *which*
          classes get confused with each other.

        We will go much deeper into metrics (precision, recall, F1, ROC/AUC)
        in the exercises — where you will also see why accuracy can be
        outright **misleading** on imbalanced data.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Summary

        - Machine learning = learning rules from data; **supervised**,
          **unsupervised**, **semi-supervised** paradigms differ in how much
          labeling you have.
        - A model is **trained** by minimising a loss on data — that is the
          entirety of "learning".
        - **Logistic regression** = linear model + sigmoid; trained with
          **cross-entropy** and gradient descent. Simple, convex, an excellent
          baseline.
        - You implemented it **by hand in ~15 lines of numpy** — and it agrees
          with scikit-learn.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Where to go next

        - **Exercise:** `notebooks/01/introduction_logistic_regression_exercise.ipynb`
          — implement logistic regression by hand, compare against sklearn,
          then try it on the breast-cancer dataset (and discover the accuracy trap).
        - **Lecture 2:** decision trees and random forests — our first
          *non-linear* classifiers.
        """
    )
    return


if __name__ == "__main__":
    app.run()
