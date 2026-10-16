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
        - Interactive demo: how the fit **stabilises** as you draw more
          random samples

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
        _fig, _ax = _plt.subplots(figsize=(7.2, 4.4))
        _w, _b = fit3_lines[_key]
        _ax.scatter(fit3_x, fit3_y, color="#111827", s=42, zorder=5)
        _ax.plot(
            _xs, _w * _xs + _b, color=_colors[_key], lw=2.5,
            label=f"your line ({_key})",
        )
        _ax.vlines(fit3_x, fit3_y, _w * fit3_x + _b, color="gray", lw=1.2)
        _ax.set_xlabel("feature $x$")
        _ax.set_ylabel("target $y$")
        _ax.set_title("Your line and its gaps to the data")
        _ax.legend(fontsize=9, loc="upper left")

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
        _width = "680px"

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
    gd_step = mo.ui.slider(
        start=0, stop=200, step=1, value=0,
        label="gradient-descent step", show_value=True, debounce=True,
    )
    mo.md(
        r"""
        ## Two ways to find the best line

        **1. The closed form.** The MSE is a smooth bowl, so its lowest point
        can be found with a formula. For one feature the best line is

        $$w = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sum_i (x_i - \bar{x})^2},
        \qquad b = \bar{y} - w\,\bar{x}$$

        where $\bar{x}$ and $\bar{y}$ are the means.

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
    return (gd_step,)


@app.cell
def _(gd_step, mo):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.linear_model import LinearRegression as _LinReg

    _rng = _np.random.default_rng(1)
    _X = _rng.normal(size=(120, 1))
    _y = 3.0 * _X.ravel() + 1.0 + _rng.normal(scale=0.7, size=120)

    _closed = _LinReg().fit(_X, _y)
    _cw = float(_closed.coef_[0])
    _cb = float(_closed.intercept_)

    _n = len(_y)
    _lr = 0.1
    _w = _b = 0.0
    _ws, _bs = [_w], [_b]
    _losses = [float(_np.mean((_X.ravel() * _w + _b - _y) ** 2))]
    for _ in range(200):
        _pred = _X.ravel() * _w + _b
        _gw = (2 / _n) * _np.sum((_pred - _y) * _X.ravel())
        _gb = (2 / _n) * _np.sum(_pred - _y)
        _w -= _lr * _gw
        _b -= _lr * _gb
        _ws.append(_w)
        _bs.append(_b)
        _losses.append(float(_np.mean((_X.ravel() * _w + _b - _y) ** 2)))

    _t = int(gd_step.value)
    _wt, _bt, _lt = _ws[_t], _bs[_t], _losses[_t]
    _xs = _np.linspace(_X.min() - 0.3, _X.max() + 0.3, 100)

    _fig, _axes = _plt.subplots(1, 3, figsize=(13.5, 4.1))

    _axes[0].scatter(_X, _y, color="#2563eb", s=16, alpha=0.55)
    _axes[0].plot(_xs, _cw * _xs + _cb, color="#dc2626", lw=2.5)
    _axes[0].set_title(rf"closed form: $\hat y = {_cw:.2f}x {_cb:+.2f}$")
    _axes[0].set_xlabel("$x$")
    _axes[0].set_ylabel("$y$")

    _axes[1].scatter(_X, _y, color="#2563eb", s=16, alpha=0.55)
    _axes[1].plot(_xs, _cw * _xs + _cb, color="#dc2626", lw=2,
                  label="closed form")
    _axes[1].plot(_xs, _wt * _xs + _bt, color="#16a34a", lw=2, ls="--",
                  label=f"gradient descent (step {_t})")
    _axes[1].set_title("both lines")
    _axes[1].set_xlabel("$x$")
    _axes[1].legend(fontsize=8, loc="upper left")

    _axes[2].plot(_losses, color="#2563eb", lw=1.8)
    _axes[2].scatter([_t], [_lt], color="#16a34a", s=45, zorder=5)
    _axes[2].set_title(f"MSE = {_lt:.3f}")
    _axes[2].set_xlabel("step")
    _axes[2].set_ylabel("MSE")

    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.hstack([gd_step], justify="start"),
            mo.image(_buf, width="1040px"),
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
    k_slider = mo.ui.slider(
        start=2, stop=int(fit_x.size), step=1, value=4,
        label="random samples (k)", show_value=True, debounce=True,
    )
    redraw = mo.ui.button(
        value=0, on_click=lambda _v: _v + 1, label="🎲 draw a new sample",
    )
    mo.md(
        r"""
        ## Interactive — fitting a line from random samples

        Each time you **draw**, we pick $k$ points **at random** (not the first
        $k$) and fit the line to those $k$ alone.

        - with **few** points the line jumps around — every draw gives a very
          different fit
        - as $k$ grows the line **settles down** towards the least-squares line
          for the whole dataset
        - drag $k$ and click **draw** a few times to feel the difference

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">11 / 16</div>
        """
    )
    return fit_x, fit_y, k_slider, redraw


@app.cell
def _(fit_x, fit_y, k_slider, mo, redraw):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.linear_model import LinearRegression as _LinReg

    _k = int(k_slider.value)
    _rng = _np.random.default_rng(int(redraw.value))
    _idx = _rng.choice(fit_x.size, size=_k, replace=False)
    _sel = _np.zeros(fit_x.size, dtype=bool)
    _sel[_idx] = True

    _full = _LinReg().fit(fit_x.reshape(-1, 1), fit_y)
    _part = _LinReg().fit(fit_x[_sel].reshape(-1, 1), fit_y[_sel])
    _xs = _np.linspace(-0.3, 10.3, 100).reshape(-1, 1)

    _fig, _ax = _plt.subplots(figsize=(7.2, 4.2))
    _ax.scatter(fit_x[~_sel], fit_y[~_sel], color="#cbd5e1", s=28, label="not drawn")
    _ax.scatter(fit_x[_sel], fit_y[_sel], color="#2563eb", s=40, label=f"drawn ({_k})")
    _ax.plot(_xs, _part.predict(_xs), color="#dc2626", lw=2.5,
             label=f"fit on {_k} samples")
    _ax.plot(_xs, _full.predict(_xs), color="#16a34a", lw=1.6, ls="--",
             label="fit on all points")
    _mse_part = _np.mean((_part.predict(fit_x[_sel].reshape(-1, 1)) - fit_y[_sel]) ** 2)
    _ax.set_xlabel("feature $x$")
    _ax.set_ylabel("target $y$")
    _ax.set_title(
        f"{_k} random samples → w = {_part.coef_[0]:.2f}, b = {_part.intercept_:+.2f}"
        f"  (MSE {_mse_part:.2f})"
    )
    _ax.legend(loc="lower right", fontsize=8)
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.hstack([k_slider, redraw], justify="start", gap=1.5),
            mo.image(_buf, width="680px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">12 / 16</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    line_w = mo.ui.slider(
        start=0, stop=2000, step=50, value=500,
        label="slope w", show_value=True, debounce=True,
    )
    line_b = mo.ui.slider(
        start=-100, stop=400, step=10, value=100,
        label="intercept b", show_value=True, debounce=True,
    )
    mo.md(
        r"""
        ## Evaluating a regression model

        - **R² (coefficient of determination)** — the fraction of the target
          variance the model explains: $1$ is perfect, $0$ means "no better than
          predicting the mean", and it *can go negative* on a bad model.
        - **RMSE** — root mean squared error, in the same units as $y$, so its
          magnitude is directly interpretable.

        Both numbers are properties of **the line you pick**. Drag the sliders
        below (diabetes: BMI → disease progression) and watch $R^2$ and MSE
        respond — the *best* line is the one that makes the MSE smallest. Report
        these on a **held-out test set**, never the training set.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">13 / 16</div>
        """
    )
    return line_b, line_w


@app.cell
def _(line_b, line_w, mo):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.datasets import load_diabetes as _load_diabetes
    from sklearn.linear_model import LinearRegression as _LinReg

    _diab = _load_diabetes()
    _x = _diab.data[:, 2]  # BMI (standardised)
    _y = _diab.target

    _best = _LinReg().fit(_x.reshape(-1, 1), _y)
    _bw = float(_best.coef_[0])
    _bb = float(_best.intercept_)

    _w = float(line_w.value)
    _b = float(line_b.value)
    _pred = _w * _x + _b
    _sst = float(((_y - _y.mean()) ** 2).sum())
    _mse_cur = float(_np.mean((_pred - _y) ** 2))
    _r2_cur = float(1 - ((_pred - _y) ** 2).sum() / _sst)
    _pred_best = _bw * _x + _bb
    _mse_best = float(_np.mean((_pred_best - _y) ** 2))
    _r2_best = float(1 - ((_pred_best - _y) ** 2).sum() / _sst)

    _xo = _np.linspace(_x.min(), _x.max(), 100)
    _fig, _axes = _plt.subplots(1, 2, figsize=(12, 4.3))

    _axes[0].scatter(_x, _y, color="#2563eb", s=18, alpha=0.6)
    _axes[0].vlines(_x, _y, _pred, color="gray", lw=0.6, alpha=0.5)
    _axes[0].plot(_xo, _w * _xo + _b, color="#dc2626", lw=2.5, label="your line")
    _axes[0].plot(_xo, _bw * _xo + _bb, color="#16a34a", lw=1.8, ls="--",
                  label="best line")
    _axes[0].set_xlabel("BMI (standardised)")
    _axes[0].set_ylabel("disease progression")
    _axes[0].set_title("drag the line around the data")
    _axes[0].legend(fontsize=8, loc="upper left")

    _axes[1].bar(
        ["your line", "best line"], [_mse_cur, _mse_best],
        color=["#dc2626", "#16a34a"],
    )
    _axes[1].text(0, _mse_cur, f"MSE {_mse_cur:.0f}\n$R^2$ {_r2_cur:.2f}",
                  ha="center", va="bottom", fontsize=9)
    _axes[1].text(1, _mse_best, f"MSE {_mse_best:.0f}\n$R^2$ {_r2_best:.2f}",
                  ha="center", va="bottom", fontsize=9)
    _axes[1].set_ylabel("MSE (lower is better)")
    _axes[1].set_title("the best line minimises the MSE")

    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.hstack([line_w, line_b], justify="start", gap=1.5),
            mo.image(_buf, width="900px"),
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
        - We can watch the fit **stabilise as we draw more random samples** —
          few points give a shaky line, many give a steady one.
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
