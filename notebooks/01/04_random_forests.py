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
# Lecture 1 (Oct 19), Session 4 — Random forests and ensembles.
# Run locally with `marimo edit notebooks/01/04_random_forests.py`
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
    layout_file="layouts/04_random_forests.slides.json",
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
        # Random Forests & Ensembles

        **Machine Learning with Python** — Lecture 1, Session 4 (Oct 19)

        EUGLOH — *Problem Solving Using Open-Source Languages; R and Python*

        University of Novi Sad

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">1 / 12</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Today's session

        - **Ensembles** — many weak learners make one strong learner
        - **Bagging** — bootstrap samples + random feature subsets
        - **Random forests** — decorrelated trees that vote
        - **Boosting** — the sequential alternative (and XGBoost)
        - **Feature importance** — what did the forest learn?
        - **Applications** — why trees dominate tabular data

        Session 4 of 4 today — thanks for a great day!

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">2 / 12</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Part 1 — Random forests and ensembles

        - We know single decision trees can **overfit**.
        - We could choose to **underfit** single decision trees (shallow stumps)…
        - …and **combine many of them into an ensemble**.

        A bunch of **weak learners** → one **strong learner**. This is a
        surprisingly effective idea!

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 12</div>
        """
    )
    return


@app.cell
def _(mo):
    n_trees_slider = mo.ui.slider(
        start=1, stop=100, step=1, value=25,
        label="n_estimators", show_value=True, debounce=True,
    )
    forest_depth_slider = mo.ui.slider(
        start=1, stop=12, step=1, value=5,
        label="max_depth per tree", show_value=True, debounce=True,
    )
    mo.md(
        r"""
        ## Random forests — bagging

        **Random forest** = many decorrelated trees that **vote**:

        1. **Bagging** — each tree is trained on a random *bootstrap* sample
           of the data (sampled with replacement)
        2. **Random feature subsets** — at each split, only a random subset of
           features may be considered → the trees become *different*
        3. **Aggregate** — majority vote (classification) or average (regression)

        Averaging many decorrelated trees cancels their individual errors —
        the forest is much more robust than any single tree.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 12</div>
        """
    )
    return forest_depth_slider, n_trees_slider


@app.cell
def _(forest_depth_slider, mo, n_trees_slider):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.datasets import make_moons as _make_moons
    from sklearn.ensemble import RandomForestClassifier as _RFC
    from sklearn.model_selection import train_test_split as _split

    _X, _y = _make_moons(n_samples=300, noise=0.3, random_state=0)
    _Xtr, _Xte, _ytr, _yte = _split(_X, _y, test_size=0.3, random_state=0, stratify=_y)
    _clf = _RFC(
        n_estimators=n_trees_slider.value,
        max_depth=forest_depth_slider.value,
        random_state=0,
    ).fit(_Xtr, _ytr)

    _xx, _yy = _np.meshgrid(
        _np.linspace(_X[:, 0].min() - 0.5, _X[:, 0].max() + 0.5, 250),
        _np.linspace(_X[:, 1].min() - 0.5, _X[:, 1].max() + 0.5, 250),
    )
    _Z = _clf.predict(_np.c_[_xx.ravel(), _yy.ravel()]).reshape(_xx.shape)

    _fig, _ax = _plt.subplots(figsize=(5.6, 4.6))
    _ax.contourf(_xx, _yy, _Z, alpha=0.25, cmap="RdYlGn")
    _ax.scatter(_X[_y == 0, 0], _X[_y == 0, 1], color="#dc2626", s=16)
    _ax.scatter(_X[_y == 1, 0], _X[_y == 1, 1], color="#16a34a", s=16)
    _ax.set_title(
        f"{n_trees_slider.value} trees, depth {forest_depth_slider.value} · "
        f"train {_clf.score(_Xtr, _ytr):.2f} · test {_clf.score(_Xte, _yte):.2f}"
    )
    _ax.set_aspect("equal")
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.hstack([n_trees_slider, forest_depth_slider]),
            mo.image(_buf, width="620px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 12</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Boosting — the other way to combine trees

        Bagging trains trees **in parallel**; **boosting** trains them
        **in sequence**, each fixing the previous one's mistakes:

        1. **Start simple** — train a weak model (a small tree)
        2. **Focus on mistakes** — increase the weight of misclassified samples
        3. **Train the next model** — fit another weak learner on the hard cases
        4. **Repeat** — each learner corrects its predecessors
        5. **Combine** — weighted vote (classification) / weighted sum (regression)

        **XGBoost** (eXtreme Gradient Boosting) is the polished, industrial
        version of this idea — possibly the most famous ML library of all
        time, and a Kaggle legend. Trees/boosting dominate **tabular** data
        leaderboards.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">6 / 12</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Applications

        Trees and ensembles are the workhorses of **tabular** machine
        learning — most real-world data lives in tables (Excel sheets).

        Two Kaggle examples:

        - **Otto recommender system** — predict e-commerce clicks, cart
          additions and orders. *Winning solution:* a 3-layer weighted
          ensemble of XGBoost, AdaBoost and a neural network.
        - **Santander customer satisfaction** — anonymized features. *Winning
          solution:* a 4-layer ensemble of XGBoost, AdaBoost, random forest
          and a neural network.

        Most data scientists reach for trees, boosting, ensembles and
        forests — often they outperform neural networks on tabular problems.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">7 / 12</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## What did the forest learn? Feature importance

        scikit-learn measures how much each feature **decreased impurity**,
        averaged over all the trees — a quick and useful ranking of which
        features matter.

        - **Impurity importance** (the default) can be misleading when features
          have different types or are correlated with each other.
        - **Permutation importance** is more reliable: shuffle the values of one
          feature and see how far the accuracy drops. If shuffling barely
          changes anything, that feature was not pulling its weight
          (`sklearn.inspection.permutation_importance`).

        In plain words: ask each column *"how much would the forest miss you if
        we scrambled you?"*

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">8 / 12</div>
        """
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.datasets import load_wine as _load_wine
    from sklearn.ensemble import RandomForestClassifier as _RFC

    _wine = _load_wine()
    _forest = _RFC(n_estimators=50, random_state=0).fit(_wine.data, _wine.target)
    _imp = _forest.feature_importances_
    _order = _np.argsort(_imp)

    _fig, _ax = _plt.subplots(figsize=(7.5, 5.5))
    _ax.barh(_np.array(_wine.feature_names)[_order], _imp[_order], color="#2563eb")
    _ax.set_title("Random-forest feature importance (wine dataset)")
    _ax.set_xlabel("importance")
    _fig.tight_layout()
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.image(_buf, width="860px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">9 / 12</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Summary

        - **Random forests** average many decorrelated trees (bagging + random
          feature subsets) — far more robust than a single tree.
        - **Boosting** trains trees sequentially, each fixing the last one's
          mistakes; **XGBoost** is the industrial standard.
        - Forests dominate **tabular** machine learning, and their
          **feature importance** is a useful (if rough) diagnostic.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">10 / 12</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Where to go next

        - **Exercise:** `notebooks/01/04_random_forests_{beginner,intermediate,advanced}.ipynb`
          — explore the data, apply **random forests** with scikit-learn, and
          implement **bagging from scratch**.
        - **Tomorrow:** **neural networks** — smooth, fully non-linear
          boundaries, built from the single neuron you met today.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">11 / 12</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Thanks for today!

        See you tomorrow for **neural networks**.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">12 / 12</div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
