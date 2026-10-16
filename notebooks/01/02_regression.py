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
# Lecture 1 (Oct 19), Session 2 — Linear regression.
# Logistic regression now lives in the Day-2 neural-network decks: the
# sigmoid / logistic intuition opens Session 1, the "one neuron" bridge and
# decision boundary open Session 2, and the by-hand training and regularisation
# sit in Session 3.
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
        # Linear Regression

        **Machine Learning with Python** — Lecture 1, Session 2 (Oct 19)

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

        - **The linear model** — a straight line through the data
        - **Least squares** — the mean squared error, the closed-form fit, and
          gradient descent
        - The 1801 story of **Piazzi, Ceres and Gauss** — one of the first fits
          to data
        - Interactive demo: watching the fit update as **new samples arrive**

        Session 2 of 4 today. *(Logistic regression — the sigmoid, cross-entropy,
        training, and multi-class — now opens tomorrow's neural-network sessions.)*

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">2 / 16</div>
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

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 16</div>
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

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 16</div>
        """
    )
    return


@app.cell
def _(mo):
    import numpy as _np

    # Hand-picked measurements built so the least-squares line is EXACTLY
    # y = 1.5x + 2.0: the residuals sum to zero and are uncorrelated with x.
    # Line C below is therefore the answer the computer will find later.
    fit3_x = _np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=float)
    fit3_y = _np.array(
        [3.0, 5.4, 5.3, 8.8, 9.0, 12.0, 13.5, 13.5, 16.3, 15.8, 18.9, 19.5]
    )
    # Three candidate lines y = w x + b, with deliberately neutral labels.
    fit3_lines = {"A": (2.0, 2.0), "B": (1.5, 5.0), "C": (1.5, 2.0)}

    fit3_pick = mo.ui.radio(
        options=["Line A", "Line B", "Line C"],
        value="Line A",
        label="Which line follows the data best?",
    )
    fit3_reveal = mo.ui.run_button(label="Reveal the verdict")
    mo.md(
        r"""
        ## Your turn — which line fits?

        Here are **12 measurements** and **three straight lines**. Two of them
        miss the data; one follows it. **No formulas yet** — judge by eye,
        commit to a line, then reveal the verdict.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 16</div>
        """
    )
    return fit3_lines, fit3_pick, fit3_reveal, fit3_x, fit3_y


@app.cell
def _(fit3_lines, fit3_pick, fit3_reveal, fit3_x, fit3_y, mo):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np

    _key = fit3_pick.value.split()[-1]
    _revealed = bool(fit3_reveal.value)
    # Colourblind-safe palette (Okabe-Ito amber / blue / bluish-green).
    _colors = {"A": "#E69F00", "B": "#0072B2", "C": "#009E73"}
    _xs = _np.linspace(0.0, 13.0, 200)
    # Sum of squared gaps for each line. Label C's value is the smallest.
    _ssg = {
        _k: float(_np.sum((_w * fit3_x + _b - fit3_y) ** 2))
        for _k, (_w, _b) in fit3_lines.items()
    }
    _best = min(_ssg, key=_ssg.get)

    if not _revealed:
        _fig, _ax = _plt.subplots(figsize=(7.2, 4.4))
        _ax.scatter(fit3_x, fit3_y, color="#111827", s=42, zorder=5, label="data")
        for _k, (_w, _b) in fit3_lines.items():
            _hot = _k == _key
            _ax.plot(
                _xs, _w * _xs + _b, color=_colors[_k],
                lw=3.2 if _hot else 1.4, alpha=1.0 if _hot else 0.35,
                label=f"Line {_k}",
            )
        _ax.set_xlabel("feature $x$")
        _ax.set_ylabel("target $y$")
        _ax.set_title("Three candidate lines — which one follows the data?")
        _ax.legend(fontsize=9, loc="upper left")
        _feedback = mo.md(
            f"**You picked Line {_key}** (drawn thicker). "
            "Press **Reveal the verdict** to check."
        )
        _width = "680px"
    else:
        _fig, _axes = _plt.subplots(1, 2, figsize=(12, 4.4))
        _w, _b = fit3_lines[_key]
        _axes[0].scatter(fit3_x, fit3_y, color="#111827", s=42, zorder=5)
        _axes[0].plot(
            _xs, _w * _xs + _b, color=_colors[_key], lw=2.5,
            label=f"your line ({_key})",
        )
        _axes[0].vlines(fit3_x, fit3_y, _w * fit3_x + _b, color="gray", lw=1.2)
        _axes[0].set_xlabel("feature $x$")
        _axes[0].set_ylabel("target $y$")
        _axes[0].set_title("Your line and its gaps to the data")
        _axes[0].legend(fontsize=9, loc="upper left")

        _order = ["A", "B", "C"]
        _axes[1].bar(
            _order, [_ssg[_k] for _k in _order], color=[_colors[_k] for _k in _order]
        )
        for _i, _k in enumerate(_order):
            _axes[1].text(_i, _ssg[_k], f"{_ssg[_k]:.1f}", ha="center", va="bottom")
        _axes[1].set_ylabel("sum of squared gaps")
        _axes[1].set_title("lower is better")

        _ok = (
            "✅ Correct — you picked the line that follows the data."
            if _key == _best
            else f"Not quite — **Line {_best}** follows the data best."
        )
        _feedback = mo.md(
            f"{_ok}\n\n"
            "We just measured “fit” by the **total squared gap** between the line "
            "and the data. On the next slide the computer minimises exactly this "
            "gap for you — and then we give it a name."
        )
        _width = "900px"

    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.hstack([fit3_pick, fit3_reveal], justify="start", gap=1.5),
            mo.image(_buf, width=_width),
            _feedback,
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">6 / 16</div>"""),
        ]
    )
    return


@app.cell
def _(fit3_x, fit3_y, mo):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.linear_model import LinearRegression as _LinReg

    _model = _LinReg().fit(fit3_x.reshape(-1, 1), fit3_y)
    _xs = _np.linspace(0.0, 13.0, 100)
    _yh = _model.predict(_xs.reshape(-1, 1))
    _pred = _model.predict(fit3_x.reshape(-1, 1))

    _fig, _ax = _plt.subplots(figsize=(7, 4))
    _ax.scatter(fit3_x, fit3_y, color="#111827", s=32, label="data")
    _ax.plot(_xs, _yh, color="#009E73", lw=2.5, label="least-squares line")
    for _xi, _yi, _pi in zip(fit3_x, fit3_y, _pred):
        _ax.plot([_xi, _xi], [_yi, _pi], color="gray", lw=1.2)
    _ax.set_xlabel("feature $x$")
    _ax.set_ylabel("target $y$")
    _ax.set_title(
        rf"the computer's fit: $\hat{{y}} = {_model.coef_[0]:.2f}x {_model.intercept_:+.2f}$"
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
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">7 / 16</div>"""),
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

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">8 / 16</div>
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

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">9 / 16</div>
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
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">10 / 16</div>"""),
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

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">11 / 16</div>
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
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">12 / 16</div>"""),
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

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">13 / 16</div>
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
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">14 / 16</div>"""),
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
          in the neural-network sessions tomorrow.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">15 / 16</div>
        """
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
        - We can watch the fit **improve as more samples arrive** — the estimate
          stabilises as the data grows.
        - **Always report on a held-out set**; use **R²** and **RMSE** to judge a
          regression model.
        - **Feature scaling** and **regularisation** are the practical levers.

        Next session: **decision trees**. Tomorrow: **logistic regression**, the
        bridge into neural networks.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">16 / 16</div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
