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
# Lecture 1 (Oct 19), Session 2 — Linear and logistic regression.
# Structure of the logistic-regression part follows the FYS-2021 slide decks
# (05_LogisticRegression / 05_LogisticRegression+Accuracy).
# Run locally with `marimo edit notebooks/01/02_regression.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).
#
# NOTE on scoping: Marimo requires each global name to be owned by exactly
# one cell. All cell-locals are underscore-prefixed. UI elements are created
# in one cell (whose *output* is the intro text — the md must be the last
# expression) and *read* in the following cell, which displays the widget
# together with its figure via mo.vstack. Figures are rendered with
# mo.image(BytesIO) — mo.as_html does not work in the Pyodide/WASM build.

import marimo

__generated_with = "0.17.6"
app = marimo.App(
    width="medium",
    layout_file="layouts/02_regression.slides.json",
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
        # Linear & Logistic Regression

        **Machine Learning with Python** — Lecture 1, Session 2 (Oct 19)

        EUGLOH — *Problem Solving Using Open-Source Languages; R and Python*

        University of Novi Sad

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">1 / 35</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Today's session

        - **Linear regression** — the linear model, least squares, closed form
          and gradient descent — and the 1801 story of **Piazzi, Ceres and
          Gauss**, one of the first fits to data
        - **Logistic regression** — from regression to classification, sigmoid,
          cross-entropy, implemented **by hand in numpy**
        - Interactive demos: watching the model update as **new samples
          arrive**, class separation, the decision threshold
        - **Multi-class** classification via one-vs-rest / softmax

        Session 2 of 4 today.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">2 / 35</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Part 1 — Linear regression

        ## The linear model

        Our first **supervised** model. With **one feature** it is simply a
        straight line:

        $$\hat{y} = w\,x + b$$

        - $w$ — the **weight** (the slope): how much the prediction changes
          when $x$ goes up by 1
        - $b$ — the **bias** (the intercept): the prediction when $x = 0$
        - $\hat{y}$ ("y-hat") is our **prediction** — a continuous number
          (a price, a temperature, …)

        With several features the same idea gives a **flat surface** (a
        *hyperplane*). We keep the one-feature picture, because it is the
        easiest to see.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 35</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The first great fit — Piazzi, Ceres and Gauss (1801)

        - On **1 January 1801** Giuseppe Piazzi, an astronomer in Palermo,
          spotted a faint moving point of light — a new body we now call the
          dwarf planet **Ceres**.
        - He tracked it for **41 nights**, then it vanished into the glare of
          the Sun. Nobody could find it again.
        - **Carl Friedrich Gauss**, then 24, took Piazzi's handful of
          observations, fitted a model to them with the method of **least
          squares**, and predicted where Ceres would reappear.
        - Astronomers searched — and found Ceres almost exactly where Gauss had
          said it would be.

        This is often called one of the **first uses of learning from data**:
        fit a model to the examples you have, then predict the ones you have
        not seen. The tool was the same **least-squares line** we are about to
        derive.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 35</div>
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
    _x = _np.linspace(0, 10, 40)
    _y = 2.0 + 1.5 * _x + _rng.normal(scale=2.5, size=_x.size)

    _model = _LinReg().fit(_x.reshape(-1, 1), _y)
    _xs = _np.linspace(-0.5, 10.5, 100)
    _yh = _model.predict(_xs.reshape(-1, 1))
    _pred = _model.predict(_x.reshape(-1, 1))

    _fig, _ax = _plt.subplots(figsize=(7, 4))
    _ax.scatter(_x, _y, color="#2563eb", s=28, label="data")
    _ax.plot(_xs, _yh, color="#dc2626", lw=2, label="fitted line")
    for _xi, _yi, _pi in zip(_x[::4], _y[::4], _pred[::4]):
        _ax.plot([_xi, _xi], [_yi, _pi], color="gray", lw=1)
    _ax.set_xlabel("feature $x$")
    _ax.set_ylabel("target $y$")
    _ax.set_title(
        rf"$\hat{{y}} = {_model.coef_[0]:.2f}x {_model.intercept_:+.2f}$"
        "  (grey = residuals)"
    )
    _ax.legend()
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.image(_buf, width="620px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 35</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Choosing the line — the mean squared error

        We want the line that is *closest* to the data. "Closest" means making
        the errors small, and we measure them with the **mean squared error**
        (MSE):

        $$\mathcal{L}(w, b) = \frac{1}{n}\sum_{i=1}^{n} (\hat{y}_i - y_i)^2$$

        In plain words: for each point take its vertical distance to the line
        (the **residual** $\hat{y}_i - y_i$), **square** it, and **average**
        over the $n$ points.

        - squaring counts every error positively and punishes large errors much
          more than small ones
        - the squared error is **smooth** and **convex** — it has one lowest
          point, so there are no local traps
        - the grey segments in the figure above are exactly those residuals

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">6 / 35</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Two ways to find the best line

        **1. The closed form.** The MSE is a smooth bowl, so its lowest point
        can be found with a formula. For one feature the best line is

        $$w = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sum_i (x_i - \bar{x})^2},
        \qquad b = \bar{y} - w\,\bar{x}$$

        where $\bar{x}$ and $\bar{y}$ are the means. No iteration — one
        calculation gives the best line. (With many features the same idea is
        written with matrices and solved once; `LinearRegression` uses a
        numerically safer version of it.)

        **2. Gradient descent.** With very many features that formula is
        expensive, so instead we *walk downhill* on the loss:

        $$w \leftarrow w - \eta \, \frac{\partial \mathcal{L}}{\partial w},
        \qquad b \leftarrow b - \eta \, \frac{\partial \mathcal{L}}{\partial b}$$

        - $\partial \mathcal{L}/\partial w$ points in the direction the loss
          grows; we step the **opposite** way
        - $\eta$ is the **learning rate** — how large each step is
        - repeat many times, and the line creeps towards the same answer

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">7 / 35</div>
        """
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.linear_model import LinearRegression as _LinReg

    _rng = _np.random.default_rng(1)
    _X = _rng.normal(size=(120, 1))
    _y = 3.0 * _X.ravel() + 1.0 + _rng.normal(scale=0.7, size=120)

    _closed = _LinReg().fit(_X, _y)

    _w, _b, _lr, _losses = _np.zeros(1), 0.0, 0.1, []
    _n = len(_y)
    for _ in range(200):
        _pred = _X @ _w + _b
        _losses.append(_np.mean((_pred - _y) ** 2))
        _gw = (2 / _n) * _X.T @ (_pred - _y)
        _gb = (2 / _n) * _np.sum(_pred - _y)
        _w -= _lr * _gw
        _b -= _lr * _gb

    _fig, _axes = _plt.subplots(1, 2, figsize=(11, 4))
    _xs = _np.linspace(_X.min() - 0.3, _X.max() + 0.3, 100).reshape(-1, 1)
    _axes[0].scatter(_X, _y, color="#2563eb", s=20, alpha=0.7)
    _axes[0].plot(_xs, _closed.predict(_xs), color="#dc2626", lw=2, label="closed form")
    _axes[0].plot(_xs, _xs @ _w + _b, color="#16a34a", lw=2, ls="--", label="gradient descent")
    _axes[0].set_title("Both methods find the same line")
    _axes[0].legend()
    _axes[1].plot(_losses, color="#2563eb")
    _axes[1].set_xlabel("epoch")
    _axes[1].set_ylabel("MSE")
    _axes[1].set_title("Gradient descent converges")
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.image(_buf, width="860px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">8 / 35</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    import numpy as _np

    _rng = _np.random.default_rng(7)
    fit_x = _np.sort(_rng.uniform(0, 10, 30))
    fit_y = 1.5 + 1.2 * fit_x + _rng.normal(scale=2.2, size=fit_x.size)
    n_seen = mo.ui.slider(
        start=2, stop=int(fit_x.size), step=1, value=4,
        label="samples seen", show_value=True, debounce=True,
    )
    mo.md(
        r"""
        ## Interactive — fitting a line as the data arrives

        We never get all the data at once. The slider reveals the first $k$
        samples; the line is re-fitted from those $k$ points **alone**.

        - with **few** points the line is shaky and its estimate jumps around
        - as $k$ grows the line **settles down** towards the least-squares
          answer for the full dataset
        - nothing new to learn here — it is the same least-squares fit, just
          computed on a growing prefix of the data

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">9 / 35</div>
        """
    )
    return fit_x, fit_y, n_seen


@app.cell
def _(fit_x, fit_y, mo, n_seen):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.linear_model import LinearRegression as _LinReg

    _k = int(n_seen.value)
    _full = _LinReg().fit(fit_x.reshape(-1, 1), fit_y)
    _part = _LinReg().fit(fit_x[:_k].reshape(-1, 1), fit_y[:_k])
    _xs = _np.linspace(-0.3, 10.3, 100).reshape(-1, 1)

    _fig, _ax = _plt.subplots(figsize=(7, 4))
    _ax.scatter(fit_x[_k:], fit_y[_k:], color="#cbd5e1", s=28, label="not yet seen")
    _ax.scatter(fit_x[:_k], fit_y[:_k], color="#2563eb", s=34, label=f"seen ({_k})")
    _ax.plot(_xs, _part.predict(_xs), color="#dc2626", lw=2.5,
             label=f"fit on {_k} points")
    _ax.plot(_xs, _full.predict(_xs), color="#16a34a", lw=1.6, ls="--",
             label="fit on all points")
    _mse_part = _np.mean((_part.predict(fit_x[:_k].reshape(-1, 1)) - fit_y[:_k]) ** 2)
    _ax.set_xlabel("feature $x$")
    _ax.set_ylabel("target $y$")
    _ax.set_title(
        f"{_k} samples → w = {_part.coef_[0]:.2f}, b = {_part.intercept_:+.2f}"
        f"  (MSE {_mse_part:.2f})"
    )
    _ax.legend(loc="lower right", fontsize=8)
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            n_seen,
            mo.image(_buf, width="680px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">10 / 35</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Evaluating a regression model

        - **R² (coefficient of determination)** — the fraction of the target
          variance the model explains: $1$ is perfect, $0$ means "no better than
          predicting the mean", and it *can go negative* on a bad model.
        - **RMSE** — root mean squared error, in the same units as $y$, so its
          magnitude is directly interpretable.

        Report every number on a **held-out test set**, never on the training
        set. Below: the diabetes dataset (age, BMI, blood pressure, …).

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">11 / 35</div>
        """
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.datasets import load_diabetes as _load_diabetes
    from sklearn.linear_model import LinearRegression as _LinReg
    from sklearn.metrics import mean_squared_error as _mse
    from sklearn.metrics import r2_score as _r2
    from sklearn.model_selection import train_test_split as _split

    _diab = _load_diabetes()
    _Xtr, _Xte, _ytr, _yte = _split(
        _diab.data, _diab.target, test_size=0.3, random_state=0
    )
    _m = _LinReg().fit(_Xtr, _ytr)
    _pred = _m.predict(_Xte)

    _rmse_model = _np.sqrt(_mse(_yte, _pred))
    _rmse_base = _np.sqrt(_mse(_yte, _np.full_like(_yte, _yte.mean())))

    _fig, _axes = _plt.subplots(1, 2, figsize=(11, 4.2))
    _axes[0].scatter(_yte, _pred, color="#2563eb", s=22, alpha=0.7)
    _lim = [_yte.min(), _yte.max()]
    _axes[0].plot(_lim, _lim, color="gray", ls="--")
    _axes[0].set_xlabel("true value")
    _axes[0].set_ylabel("predicted value")
    _axes[0].set_title(
        f"closer to the dashed line = better (test $R^2$ = {_r2(_yte, _pred):.2f})"
    )
    _axes[1].bar(
        ["predict the mean\n(baseline)", "our model"],
        [_rmse_base, _rmse_model],
        color=["#94a3b8", "#16a34a"],
    )
    _axes[1].set_ylabel("RMSE (same units as the target)")
    _axes[1].set_title("lower is better")
    for _i, _v in enumerate([_rmse_base, _rmse_model]):
        _axes[1].text(_i, _v, f"{_v:.0f}", ha="center", va="bottom")
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.image(_buf, width="860px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">12 / 35</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Linear regression in scikit-learn

        ```python
        from sklearn.linear_model import LinearRegression, SGDRegressor
        model = LinearRegression().fit(X_train, y_train)   # closed form
        model.score(X_test, y_test)                         # R²
        ```

        - `LinearRegression` — exact least squares (closed form)
        - `SGDRegressor` — gradient descent; scales to very large datasets
        - **regularised** cousins: `Ridge` (L2 penalty) and `Lasso` (L1) trade a
          little bias for a lot less variance — the same idea we will meet again
          in logistic regression.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">13 / 35</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Part 2 — Logistic regression

        ## From regression to classification

        Linear regression predicts a **number**. But many problems ask for a
        **label** (spam / not spam, healthy / diseased).

        Feed $0/1$ labels into a line and it happily predicts $-0.4$ or $1.4$ —
        which is meaningless as a probability:

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">14 / 35</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Recap: the linear model

        Linear regression predicts a **continuous** number with a straight line
        $\hat{y} = w x + b$, fitted by making the **mean squared error** small:

        $$\mathcal{L}(w, b) = \frac{1}{n} \sum_i (\hat{y}_i - y_i)^2$$

        We improve the line step by step, nudging each parameter *against* the
        direction in which the loss grows:

        $$w \leftarrow w - \eta \, \frac{\partial \mathcal{L}}{\partial w}$$

        — this is **gradient descent**, and $\eta$ is the **learning rate**.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">15 / 35</div>
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
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">16 / 35</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The logistic function

        For classification the labels take only **two values** (0 and 1), so we
        want a function bounded between 0 and 1 with a sharp transition — the
        **logistic (sigmoid) function**:

        $$\sigma(z) = \frac{1}{1 + e^{-z}}$$

        In plain words: feed in **any** number $z$ and $\sigma$ bends it into
        something between $0$ and $1$. A large positive $z$ gives almost $1$;
        a large negative $z$ gives almost $0$; and $z = 0$ gives exactly
        $0.5$.

        It is smooth everywhere, which is what lets us train it with gradient
        descent.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">17 / 35</div>
        """
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np

    _z = _np.linspace(-6, 6, 300)
    _sig = 1.0 / (1.0 + _np.exp(-_z))

    _fig, _ax = _plt.subplots(figsize=(7, 3.4))
    _ax.plot(_z, _sig, color="#2563eb", linewidth=2.5)
    _ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    _ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    _ax.set_xlabel("z")
    _ax.set_ylabel("σ(z)")
    _ax.set_title("Bounded between 0 and 1, with a sharp transition")
    _ax.grid(alpha=0.3)
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.image(_buf, width="620px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">18 / 35</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## A first intuition: compress the line into [0, 1]

        Take the linear model $z = w^\top x + b$, then **squeeze** its output
        through the sigmoid: $x \rightarrow w^\top x + b \rightarrow \sigma(w^\top x + b)$.

        - far left: $\sigma \approx 0$ — confidently class 0
        - far right: $\sigma \approx 1$ — confidently class 1
        - in between: a probability!

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">19 / 35</div>
        """
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.linear_model import LinearRegression as _LinReg
    from sklearn.linear_model import LogisticRegression as _LR

    _rng = _np.random.default_rng(0)
    _x0 = _rng.normal(loc=2.0, scale=0.9, size=60)
    _x1 = _rng.normal(loc=6.0, scale=0.9, size=60)
    _x = _np.concatenate([_x0, _x1])
    _y = _np.concatenate([_np.zeros(60), _np.ones(60)])
    _xs = _np.linspace(0, 8, 300)

    _lin = _LinReg().fit(_x.reshape(-1, 1), _y)
    _log = _LR().fit(_x.reshape(-1, 1), _y)

    _fig, _axes = _plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    _axes[0].scatter(_x, _y, c=_y, cmap="RdYlGn", s=22)
    _axes[0].plot(_xs, _lin.predict(_xs.reshape(-1, 1)), color="#2563eb", lw=2)
    _axes[0].set_title("1. fit a line (unbounded)")
    _axes[1].scatter(_x, _y, c=_y, cmap="RdYlGn", s=22)
    _axes[1].plot(_xs, _log.predict_proba(_xs.reshape(-1, 1))[:, 1], color="#dc2626", lw=2)
    _axes[1].axhline(0.5, color="gray", ls="--", lw=0.8)
    _axes[1].set_title("2. squeeze it through σ → probabilities in [0, 1]")
    for _ax in _axes:
        _ax.set_xlabel("feature $x$")
    _axes[0].set_ylabel("label / probability")
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.image(_buf, width="860px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">20 / 35</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The statistical view

        Encode the classes as numbers: class 1 → $y = 1$, class 2 → $y = 0$.
        The sigmoid output is then read as the **probability of class 1** given
        the input.

        **Why a sigmoid?** Picture the two classes as two **bell curves**
        (Gaussians) with the *same spread*. The probability of class 1 then
        works out to be exactly a logistic function of a linear score.

        **Summary:** the model predicts $\hat{p} = \sigma(z)$ with
        $z = w x + b$. At $z = 0$ the probability is $0.5$ — that is the
        **decision boundary**, a straight line (a *hyperplane* in higher
        dimensions). Logistic regression does not fit the points; it finds the
        **separation** between the classes.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">21 / 35</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Learning: the cross-entropy loss

        We want the model to give a **high probability to the true label**, and
        to pay a price when it is confidently wrong. The **cross-entropy**
        does exactly that:

        $$\mathcal{L}(w, b) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log \hat{p}_i + (1 - y_i) \log (1 - \hat{p}_i) \right]$$

        - small when the prediction matches the true label, large when it is
          confidently wrong
        - **convex** — gradient descent finds the single global minimum

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">22 / 35</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Minimising the cross-entropy

        Setting the derivative to zero (as we did for linear regression) does
        **not** give a formula for $w$ — there is **no closed-form solution**.
        But the gradient is remarkably simple:

        $$\frac{\partial \mathcal{L}}{\partial w} = \frac{1}{n}\sum_i (\hat{p}_i - y_i)\, x_i,
        \qquad \frac{\partial \mathcal{L}}{\partial b} = \frac{1}{n}\sum_i (\hat{p}_i - y_i)$$

        - the "error" here is just $\hat{p}_i - y_i$: predicted probability
          minus true label
        - the gradient is **small when the predictions are good** — learning
          slows down on its own
        - it is almost the same rule as for linear regression, with $\hat{p}$
          in place of $\hat{y}$

        Repeated many times (each full pass over the data is an **epoch**):

        $$w \leftarrow w - \eta \, \frac{\partial \mathcal{L}}{\partial w}$$

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">23 / 35</div>
        """
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

        Keep this picture — it is the seed of tomorrow's lecture on neural networks.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">24 / 35</div>
        """
    )
    return


@app.cell
def _(mo):
    import numpy as _np
    from sklearn.datasets import make_blobs as _make_blobs

    def fit_logistic_gd(X, y, lr=0.5, n_iter=500):
        """Train binary logistic regression with full-batch gradient descent."""
        n, d = X.shape
        w = _np.zeros(d)
        b = 0.0
        losses = []
        for _ in range(n_iter):
            p = 1.0 / (1.0 + _np.exp(-(X @ w + b)))
            losses.append(
                -_np.mean(y * _np.log(p + 1e-12) + (1 - y) * _np.log(1 - p + 1e-12))
            )
            w -= lr * (X.T @ (p - y)) / n
            b -= lr * _np.mean(p - y)
        return w, b, losses

    blobs_X, blobs_y = _make_blobs(
        n_samples=200, centers=[(-2.0, -2.0), (2.0, 2.0)],
        cluster_std=1.2, random_state=0,
    )
    w_hand, b_hand, losses_hand = fit_logistic_gd(blobs_X, blobs_y, n_iter=500)
    acc_hand = _np.mean(
        (1.0 / (1.0 + _np.exp(-(blobs_X @ w_hand + b_hand))) >= 0.5) == blobs_y
    )

    train_slider = mo.ui.slider(
        start=1, stop=100, step=1, value=5,
        label="epochs of training", show_value=True, debounce=True,
    )
    mo.md(
        r"""
        ## Logistic regression by hand (numpy) — and in scikit-learn

        ```python
        for _ in range(n_iter):                    # epochs
            p = sigmoid(X @ w + b)                 # predict
            w -= lr * (X.T @ (p - y)) / n          # gradient step
            b -= lr * np.mean(p - y)
        ```

        ~15 lines in total. We train it on the two-blob dataset — and on the
        next slide you can **scrub through the training**, one bit at a time.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">25 / 35</div>
        """
    )
    return (
        acc_hand,
        b_hand,
        blobs_X,
        blobs_y,
        fit_logistic_gd,
        losses_hand,
        train_slider,
        w_hand,
    )


@app.cell
def _(blobs_X, blobs_y, fit_logistic_gd, mo, train_slider):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np

    _n_iter = int(train_slider.value)
    _w, _b, _losses = fit_logistic_gd(blobs_X, blobs_y, n_iter=_n_iter)
    _prob = 1.0 / (1.0 + _np.exp(-(blobs_X @ _w + _b)))
    _acc = _np.mean((_prob >= 0.5) == blobs_y)

    _xx, _yy = _np.meshgrid(
        _np.linspace(blobs_X[:, 0].min() - 1, blobs_X[:, 0].max() + 1, 250),
        _np.linspace(blobs_X[:, 1].min() - 1, blobs_X[:, 1].max() + 1, 250),
    )
    _pp = (
        1.0 / (1.0 + _np.exp(-(_np.c_[_xx.ravel(), _yy.ravel()] @ _w + _b)))
    ).reshape(_xx.shape)

    _fig, _axes = _plt.subplots(1, 2, figsize=(11, 4))
    _axes[0].plot(_losses, color="#2563eb")
    _axes[0].set_xlabel("epoch")
    _axes[0].set_ylabel("cross-entropy")
    _axes[0].set_xlim(0, 100)
    _axes[0].set_ylim(0, 0.75)
    _axes[0].set_title(f"loss after {_n_iter} epochs")
    _cs = _axes[1].contourf(_xx, _yy, _pp, levels=20, cmap="RdYlGn", alpha=0.7)
    _axes[1].scatter(blobs_X[blobs_y == 0, 0], blobs_X[blobs_y == 0, 1],
                     color="#dc2626", s=16, label="class 0")
    _axes[1].scatter(blobs_X[blobs_y == 1, 0], blobs_X[blobs_y == 1, 1],
                     color="#16a34a", s=16, label="class 1")
    _axes[1].set_title(f"boundary after {_n_iter} epochs (accuracy {_acc:.2f})")
    _axes[1].legend()
    _fig.colorbar(_cs, ax=_axes[1], label="p(class 1)")
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.md(
                r"""**Scrub through training** — watch the loss fall and the boundary rotate into place."""
            ),
            train_slider,
            mo.image(_buf, width="860px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">26 / 35</div>"""),
        ]
    )
    return


@app.cell
def _(acc_hand, b_hand, blobs_X, blobs_y, mo, w_hand):
    from sklearn.linear_model import LogisticRegression as _LR

    logreg_clf = _LR().fit(blobs_X, blobs_y)
    print(f"scikit-learn: w = {logreg_clf.coef_[0].round(3)}, "
          f"b = {logreg_clf.intercept_[0]:+.3f}, "
          f"accuracy = {logreg_clf.score(blobs_X, blobs_y):.3f}")
    print(f"by hand:      w = {w_hand.round(3)}, b = {b_hand:+.3f}, accuracy = {acc_hand:.3f}")
    print("(sklearn's weights are slightly smaller — it applies L2 regularisation by default)")
    mo.md(
        r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">27 / 35</div>"""
    )
    return (logreg_clf,)


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

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">28 / 35</div>
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
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">29 / 35</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Practical considerations & assumptions

        - **Feature scaling** — standardise features (mean 0, std 1) so gradient
          descent converges quickly and features are treated fairly.
        - **Regularisation** — sklearn penalises large weights by default
          (`C = 1/λ`, larger `C` → weaker regularisation).
        - **Probabilities vs labels** — `predict_proba` gives probabilities;
          `predict` applies the 0.5 threshold.

        Logistic regression works best when:

        - observations are **independent**,
        - the target is **binary** (otherwise: softmax),
        - there are **no strong outliers**, and the **sample size is large**.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">30 / 35</div>
        """
    )
    return


@app.cell
def _(mo):
    # The widget is created here; the *next* cell reads `.value` and displays
    # the slider together with its figure. The md must be the LAST expression
    # so it becomes the cell's output.
    sep_slider = mo.ui.slider(
        start=0.5, stop=4.0, step=0.1, value=2.0,
        label="Class separation", show_value=True, debounce=True,
    )
    mo.md(
        r"""
        ## Interactive demo — class separation

        The slider controls how far apart the two classes are. Watch the
        decision boundary and the accuracy react.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">31 / 35</div>
        """
    )
    return (sep_slider,)


@app.cell
def _(mo, sep_slider):
    import io as _io

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

    _fig, _ax = _plt.subplots(figsize=(5.6, 4.6))
    _ax.contourf(_xx, _yy, _Z, alpha=0.2, cmap="RdYlGn")
    _ax.scatter(_Xd[_yd == 1, 0], _Xd[_yd == 1, 1], color="#16a34a", alpha=0.7, label="class 1")
    _ax.scatter(_Xd[_yd == 0, 0], _Xd[_yd == 0, 1], color="#dc2626", alpha=0.7, label="class 0")
    _ax.set_xlabel("$x_1$")
    _ax.set_ylabel("$x_2$")
    _ax.set_title(f"Training accuracy: {_train_acc:.3f}")
    _ax.legend()
    _ax.set_aspect("equal")
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            sep_slider,
            mo.image(_buf, width="620px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">32 / 35</div>"""),
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

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">33 / 35</div>
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

    _fig, _axes = _plt.subplots(1, 4, figsize=(16, 4.0))
    _xs = _np.linspace(_x0, _x1, 100)
    for _k in range(len(_names)):
        _ax = _axes[_k]
        _w, _b = _ovr[_k].coef_[0], _ovr[_k].intercept_[0]
        _ax.scatter(_X[_y == _k, 0], _X[_y == _k, 1], color="#16a34a", s=20,
                    label=_names[_k])
        _ax.scatter(_X[_y != _k, 0], _X[_y != _k, 1], color="#cbd5e1", s=14,
                    label="rest")
        if abs(_w[1]) > 1e-9:
            _ax.plot(_xs, -(_w[0] * _xs + _b) / _w[1], color="#111827", lw=2)
        _ax.set_title(f"{_names[_k]} vs rest")
        _ax.set_xlabel("petal length")
        _ax.set_aspect("equal")
        _ax.legend(fontsize=7, loc="upper left")

    _axes[3].contourf(_xx, _yy, _final, alpha=0.25, cmap="RdYlGn")
    for _k in range(len(_names)):
        _axes[3].scatter(_X[_y == _k, 0], _X[_y == _k, 1], s=20, label=_names[_k])
    _axes[3].set_title("final: most confident class")
    _axes[3].set_xlabel("petal length")
    _axes[3].set_aspect("equal")
    _axes[3].legend(fontsize=7, loc="upper left")

    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.md(
                r"""Three separate **class vs rest** lines, then the region each one wins."""
            ),
            mo.image(_buf, width="980px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">34 / 35</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Summary

        - **Linear regression** predicts a number and minimises the **MSE** —
          solvable in closed form, or by **gradient descent**. (Gauss fitted
          least squares to Piazzi's Ceres observations back in 1801.)
        - **Logistic regression** = linear model + sigmoid, trained on the
          **cross-entropy**; there is no closed form, but the gradient is simple.
        - Both models can be watched **improving as more samples arrive** — the
          estimate stabilises as the data grows.
        - Logistic regression is exactly **one neuron** — the seed of the neural
          networks.
        - **Feature scaling** and **regularisation** are the practical levers;
          `predict_proba` vs `predict` is the probability/label distinction.
        - **Softmax / one-vs-rest** extend it to many classes.

        Next session: **decision trees**.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">35 / 35</div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
