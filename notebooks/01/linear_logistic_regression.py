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
# Lecture 1 — Linear and Logistic Regression.
# Run locally with `marimo edit notebooks/01/linear_logistic_regression.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).

import marimo

__generated_with = "0.17.6"
app = marimo.App(
    width="medium",
    layout_file="layouts/linear_logistic_regression.slides.json",
)


@app.cell
def _():
    import os
    from pathlib import Path

    # Make relative `media/` paths resolve the same way regardless of which
    # directory `marimo edit` is launched from. In WASM, `__file__` is not set
    # and the chdir becomes a no-op; the browser still fetches images via URL.
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
        # Linear and Logistic Regression

        **Machine Learning with Python** — Lecture 1

        EUGLOH \u2014 *Problem Solving Using Open-Source Languages; R and Python*

        University of Novi Sad
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Today's lecture

        - **Linear regression** \u2014 predict a continuous number
        - **Logistic regression** \u2014 predict a class label
        - Both are linear models \u2014 simple, fast, and surprisingly useful
        - Both have closed-form or convex objectives \u2014 well-understood maths
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Linear Regression

        ## The problem

        Given pairs $(x_i, y_i)$, find a function $f(x)$ that predicts $y$ from $x$.

        For linear regression we restrict $f$ to a **linear function**:

        $$\hat{y} = w^\top x + b$$

        where $w$ are the **weights** and $b$ is the **bias**.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The loss function

        How do we measure how good a fit is? The **mean squared error**:

        $$\mathcal{L}(w, b) = \frac{1}{n} \sum_{i=1}^{n} \left( \hat{y}_i - y_i \right)^2$$

        - Penalises large errors more than small ones
        - Differentiable \u2192 we can use gradient descent
        - Has a **closed-form solution** for linear models (the *normal equations*)
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Two ways to fit a linear model

        1. **Closed-form (normal equations)**

        $$w^* = (X^\top X)^{-1} X^\top y$$

        - Exact, but $O(d^3)$ in the number of features $d$
        - Great for small problems, painful for large ones

        2. **Gradient descent**

        $$w \leftarrow w - \eta \, \nabla_w \mathcal{L}$$

        - Iterative, scales to huge models
        - $\eta$ is the **learning rate**
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    return np, plt


@app.cell
def _(mo, np, plt):
    rng = np.random.default_rng(0)
    n = 60
    x = np.linspace(0, 10, n)
    y = 2.5 * x + 1.0 + rng.normal(scale=2.0, size=n)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x, y, color="#2563eb", alpha=0.7)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Synthetic regression dataset")
    mo.as_html(fig)
    return ax, fig, n, rng, x, y


@app.cell
def _(mo, np, plt, x, y):
    # Closed-form fit: w = (X^T X)^-1 X^T y
    X = np.column_stack([x, np.ones_like(x)])
    w, b = np.linalg.lstsq(X, y, rcond=None)[0]
    print(f"Learned slope:     {w:.3f}")
    print(f"Learned intercept: {b:.3f}")

    xs = np.linspace(0, 10, 100)
    ys = w * xs + b

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x, y, color="#2563eb", alpha=0.6, label="data")
    ax.plot(xs, ys, color="#dc2626", linewidth=2, label=f"fit: y = {w:.2f}x + {b:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Linear regression \u2014 closed-form fit")
    ax.legend()
    mo.as_html(fig)
    return X, ax, b, fig, xs, ys


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Same thing in scikit-learn

        ```python
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(x.reshape(-1, 1), y)
        ```

        \u2014 `sklearn` uses the same normal equations under the hood (or a
        numerically robust variant). For larger problems it falls back to
        coordinate descent.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Evaluating a regression model

        - **MSE / RMSE** \u2014 average squared error (and its square root)
        - **MAE** \u2014 mean absolute error; more robust to outliers
        - **$R^2$** \u2014 fraction of variance explained; $1$ is perfect, $0$ is "as good as predicting the mean"

        $$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

        Always evaluate on a **held-out test set** \u2014 not the training data.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Logistic Regression

        ## From regression to classification

        Linear regression predicts a **continuous** number. What if we want to
        predict a **class label** (e.g. spam / not spam)?

        Idea: take the linear model and squash its output through the
        **sigmoid function**:

        $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
        """
    )
    return


@app.cell
def _(mo, np, plt):
    z = np.linspace(-6, 6, 200)
    sig = 1.0 / (1.0 + np.exp(-z))

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(z, sig, color="#2563eb", linewidth=2.5)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("z = w\\u1d40x + b")
    ax.set_ylabel("\u03c3(z)")
    ax.set_title("The sigmoid function")
    ax.grid(alpha=0.3)
    mo.as_html(fig)
    return ax, fig, sig, z


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The logistic regression model

        $$\hat{p}(y=1 \mid x) = \sigma(w^\top x + b) = \frac{1}{1 + e^{-(w^\top x + b)}}$$

        - Output is a **probability** in $[0, 1]$
        - Predict class $1$ if $\hat{p} \geq 0.5$, else class $0$
        - The decision boundary is where $w^\top x + b = 0$ \u2014 a **hyperplane**
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The loss function: cross-entropy

        MSE does not work well for classification (the surface is non-convex
        in $w$ when combined with the sigmoid). Use the **log loss** instead:

        $$\mathcal{L}(w, b) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log \hat{p}_i + (1 - y_i) \log (1 - \hat{p}_i) \right]$$

        - Same optimum as the **negative log-likelihood** under a Bernoulli model
        - Convex in $w$ \u2192 gradient descent converges to a global minimum
        """
    )
    return


@app.cell
def _(mo, np, plt, rng):
    # Generate a simple 2D classification problem
    n_per = 60
    X_pos = rng.normal(loc=[2.0, 2.0], scale=[1.2, 1.2], size=(n_per, 2))
    X_neg = rng.normal(loc=[-2.0, -2.0], scale=[1.2, 1.2], size=(n_per, 2))
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(n_per), np.zeros(n_per)])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X_pos[:, 0], X_pos[:, 1], color="#16a34a", alpha=0.7, label="class 1")
    ax.scatter(X_neg[:, 0], X_neg[:, 1], color="#dc2626", alpha=0.7, label="class 0")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("2D binary classification dataset")
    ax.legend()
    ax.set_aspect("equal")
    mo.as_html(fig)
    return X, X_neg, X_pos, ax, fig, n_per, y


@app.cell
def _(mo, np, plt, X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    clf = LogisticRegression().fit(X, y)
    acc = accuracy_score(y, clf.predict(X))
    print(f"Training accuracy: {acc:.3f}")
    print(f"Learned weights:   {clf.coef_[0]}")
    print(f"Learned bias:      {clf.intercept_[0]:.3f}")

    # Plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(xx, yy, Z, alpha=0.2, cmap="RdYlGn")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], color="#16a34a", alpha=0.8, label="class 1")
    ax.scatter(X[y == 0, 0], X[y == 0, 1], color="#dc2626", alpha=0.8, label="class 0")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Logistic regression decision boundary")
    ax.legend()
    ax.set_aspect("equal")
    mo.as_html(fig)
    return (
        LogisticRegression,
        Z,
        acc,
        accuracy_score,
        ax,
        clf,
        fig,
        x_max,
        x_min,
        xx,
        y_max,
        y_min,
        yy,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ## A toy interactive demo

        The slider below controls how separated the two classes are \u2014 try it
        and watch the decision boundary move and the accuracy change.
        """
    )
    return


@app.cell
def _(mo):
    separation = mo.ui.slider(
        start=0.5, stop=4.0, step=0.1, value=2.0,
        label="Class separation",
    )
    separation
    return (separation,)


@app.cell
def _(mo, np, plt, separation):
    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.metrics import accuracy_score as _acc

    rng = np.random.default_rng(42)
    sep = separation.value
    Xp = rng.normal(loc=[sep, sep], scale=[1.0, 1.0], size=(80, 2))
    Xn = rng.normal(loc=[-sep, -sep], scale=[1.0, 1.0], size=(80, 2))
    Xd = np.vstack([Xp, Xn])
    yd = np.concatenate([np.ones(80), np.zeros(80)])
    _clf = _LR().fit(Xd, yd)
    train_acc = _acc(yd, _clf.predict(Xd))

    x0, x1 = Xd[:, 0].min() - 1, Xd[:, 0].max() + 1
    y0, y1 = Xd[:, 1].min() - 1, Xd[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x0, x1, 200), np.linspace(y0, y1, 200))
    Zd = _clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(xx, yy, Zd, alpha=0.2, cmap="RdYlGn")
    ax.scatter(Xd[yd == 1, 0], Xd[yd == 1, 1], color="#16a34a", alpha=0.7, label="class 1")
    ax.scatter(Xd[yd == 0, 0], Xd[yd == 0, 1], color="#dc2626", alpha=0.7, label="class 0")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(f"Training accuracy: {train_acc:.3f}")
    ax.legend()
    ax.set_aspect("equal")
    mo.as_html(fig)
    return (
        Xd,
        Xn,
        Xp,
        Zd,
        _LR,
        _acc,
        _clf,
        ax,
        fig,
        sep,
        train_acc,
        x0,
        x1,
        xx,
        y0,
        y1,
        yd,
        yy,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Practical considerations

        - **Feature scaling** \u2014 standardise features before fitting (mean 0, std 1). Helps gradient descent converge and is essential for regularised models.
        - **Regularisation** \u2014 add a penalty $\lambda \|w\|^2$ to the loss to prevent overfitting (`LogisticRegression(penalty="l2", C=1/\lambda)` in sklearn).
        - **Multi-class** \u2014 sklearn trains one-vs-rest by default; for mutually exclusive classes use the softmax (multinomial logistic regression).
        - **Probabilities** \u2014 `model.predict_proba(X)` gives you the calibrated class probabilities; `model.predict(X)` gives the hard label.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Summary

        - **Linear regression** \u2014 predict a continuous number; fit by minimising MSE. Closed-form or gradient descent.
        - **Logistic regression** \u2014 predict a class probability; fit by minimising cross-entropy. Convex, fast, well-calibrated.
        - Both are **linear models** \u2014 they learn a linear combination of features. The non-linearity in logistic regression comes from the sigmoid applied *after* the linear part.
        - Both are excellent **baselines** \u2014 always try them before reaching for fancier models.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Where to go next

        - Lecture 2: **Decision trees and random forests**
        - In the meantime: install the environment with `conda env create -f env.yml` and explore the `intro.ipynb` notebook
        """
    )
    return


if __name__ == "__main__":
    app.run()
