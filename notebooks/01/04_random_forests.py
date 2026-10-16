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
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np

    n_trees_slider = mo.ui.slider(
        start=1, stop=100, step=1, value=25,
        label="n_estimators", show_value=True, debounce=True,
    )
    forest_depth_slider = mo.ui.slider(
        start=1, stop=12, step=1, value=5,
        label="max_depth per tree", show_value=True, debounce=True,
    )

    # Bootstrap illustration: 12 points on a grid, then three resamples of
    # size 12 drawn *with replacement*. Colour = how many times each point was
    # drawn (grey = not drawn at all; darker = drawn more than once).
    _pts = _np.array([[i % 4, i // 4] for i in range(12)], dtype=float)
    _rng = _np.random.default_rng(0)
    _draws = [_rng.integers(0, 12, size=12) for _ in range(3)]
    _shade = ["#cbd5e1", "#93c5fd", "#2563eb", "#1e3a8a"]

    _fig, _axes = _plt.subplots(2, 2, figsize=(7.2, 5.8))
    _panels = [None, _draws[0], _draws[1], _draws[2]]
    _titles = ["all samples"] + [
        f"bootstrap {_i + 1} — {len(_np.unique(_d))} of 12 distinct"
        for _i, _d in enumerate(_draws)
    ]
    for _ax, _d, _title in zip(_axes.ravel(), _panels, _titles):
        if _d is None:
            _ax.scatter(_pts[:, 0], _pts[:, 1], s=150, color=_shade[2],
                        edgecolors="white", zorder=3)
        else:
            _counts = _np.bincount(_d, minlength=12)
            for _i in range(12):
                _ax.scatter(_pts[_i, 0], _pts[_i, 1], s=150,
                            color=_shade[min(int(_counts[_i]), 3)],
                            edgecolors="white", zorder=3)
        _ax.set_xlim(-0.7, 3.7)
        _ax.set_ylim(-0.7, 2.7)
        _ax.set_xticks([])
        _ax.set_yticks([])
        _ax.set_aspect("equal")
        _ax.set_title(_title, fontsize=10)
    _fig.suptitle(
        "bagging: each tree sees a bootstrap resample (grey = left out, "
        "darker = drawn more than once)",
        fontsize=10,
    )
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)

    _text = mo.md(
        r"""
        ## Random forests — bagging

        **Random forest** = many decorrelated trees that **vote**:

        1. **Bagging** — each tree is trained on a random *bootstrap* sample
           of the data (sampled with replacement; see the right-hand figure)
        2. **Random feature subsets** — at each split, only a random subset of
           features may be considered → the trees become *different*
        3. **Aggregate** — majority vote (classification) or average (regression)

        Averaging many decorrelated trees cancels their individual errors —
        the forest is much more robust than any single tree.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 12</div>
        """
    )
    mo.hstack(
        [_text, mo.image(_buf, width="470px")],
        widths=[1, 1], align="start", gap=1.5,
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
    from sklearn.tree import DecisionTreeClassifier as _DTC

    _X, _y = _make_moons(n_samples=1000, noise=0.3, random_state=0)
    _Xtr, _Xte, _ytr, _yte = _split(_X, _y, test_size=0.3, random_state=0, stratify=_y)

    _depth = int(forest_depth_slider.value)
    _n = int(n_trees_slider.value)
    _clf = _RFC(n_estimators=_n, max_depth=_depth, random_state=0).fit(_Xtr, _ytr)

    # Test accuracy as trees are added (warm-start grows the forest in place).
    _ns = list(range(1, 101))
    _wf = _RFC(n_estimators=1, max_depth=_depth, random_state=0, warm_start=True)
    _accs = []
    for _k in _ns:
        _wf.set_params(n_estimators=_k)
        _wf.fit(_Xtr, _ytr)
        _accs.append(_wf.score(_Xte, _yte))
    _single = _DTC(max_depth=_depth, random_state=0).fit(_Xtr, _ytr).score(_Xte, _yte)

    _xx, _yy = _np.meshgrid(
        _np.linspace(_X[:, 0].min() - 0.5, _X[:, 0].max() + 0.5, 250),
        _np.linspace(_X[:, 1].min() - 0.5, _X[:, 1].max() + 0.5, 250),
    )
    _Z = _clf.predict(_np.c_[_xx.ravel(), _yy.ravel()]).reshape(_xx.shape)

    _fig, _axes = _plt.subplots(1, 2, figsize=(12, 4.6))
    _axes[0].contourf(_xx, _yy, _Z, alpha=0.25, cmap="RdYlGn")
    _axes[0].scatter(_X[_y == 0, 0], _X[_y == 0, 1], color="#dc2626", s=14)
    _axes[0].scatter(_X[_y == 1, 0], _X[_y == 1, 1], color="#16a34a", s=14)
    _axes[0].set_title(
        f"{_n} trees, depth {_depth} · "
        f"train {_clf.score(_Xtr, _ytr):.2f} · test {_clf.score(_Xte, _yte):.2f}"
    )
    _axes[0].set_aspect("equal")

    _axes[1].plot(_ns, _accs, color="#2563eb", lw=2, label="random forest")
    _axes[1].axhline(_single, color="#dc2626", ls="--",
                     label=f"single tree (depth {_depth})")
    _axes[1].axvline(_n, color="gray", ls=":")
    _axes[1].set_xlabel("number of trees")
    _axes[1].set_ylabel("test accuracy")
    _axes[1].set_ylim(min(min(_accs), _single) - 0.03, 1.02)
    _axes[1].set_title("adding trees helps — then it plateaus")
    _axes[1].legend(fontsize=8, loc="lower right")
    _axes[1].grid(alpha=0.3)

    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.hstack([n_trees_slider, forest_depth_slider]),
            mo.image(_buf, width="920px"),
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
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.datasets import make_circles as _make_circles
    from sklearn.ensemble import AdaBoostClassifier as _AdaBoost
    from sklearn.model_selection import train_test_split as _split
    from sklearn.tree import DecisionTreeClassifier as _DTC

    _X, _y = _make_circles(n_samples=400, noise=0.15, factor=0.4, random_state=0)
    _Xtr, _Xte, _ytr, _yte = _split(_X, _y, test_size=0.3, random_state=0, stratify=_y)

    _xx, _yy = _np.meshgrid(
        _np.linspace(_X[:, 0].min() - 0.3, _X[:, 0].max() + 0.3, 250),
        _np.linspace(_X[:, 1].min() - 0.3, _X[:, 1].max() + 0.3, 250),
    )
    _stages = [1, 3, 10, 50]
    _fig, _axes = _plt.subplots(1, 4, figsize=(16, 4.3))
    for _ax, _k in zip(_axes, _stages):
        _clf = _AdaBoost(
            estimator=_DTC(max_depth=1), n_estimators=_k, random_state=0,
        ).fit(_Xtr, _ytr)
        _Z = _clf.predict(_np.c_[_xx.ravel(), _yy.ravel()]).reshape(_xx.shape)
        _ax.contourf(_xx, _yy, _Z, alpha=0.25, cmap="RdYlGn")
        _ax.scatter(_Xte[_yte == 0, 0], _Xte[_yte == 0, 1], color="#dc2626", s=14)
        _ax.scatter(_Xte[_yte == 1, 0], _Xte[_yte == 1, 1], color="#16a34a", s=14)
        _ax.set_title(
            f"{_k} stump{'s' if _k > 1 else ''} · "
            f"test {_clf.score(_Xte, _yte):.2f}"
        )
        _ax.set_aspect("equal")
        _ax.set_xticks([])
        _ax.set_yticks([])
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.md(
                r"""## Boosting in action — stumps that fix each other

                Each stage is a single **decision stump** (one axis-aligned
                cut) trained to fix what the previous stages got wrong. A
                single stump is nearly useless on this ring; a few get it
                partly right; by the time we have tens of stumps the combined
                boundary has bent into a **circle** — a shape no single stump
                could ever produce. Shown on held-out points."""
            ),
            mo.image(_buf, width="1000px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">7 / 12</div>"""),
        ]
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

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">8 / 12</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Why trees love tables — and not images

        A tree asks *"is this one feature above or below a threshold?"*. That
        is a **meaningful question only when a single feature is informative on
        its own** — exactly what a well-designed table gives you.

        - **Tabular data** — every column is *engineered*: it has a name, a
          unit and a meaning (age, income, a sensor reading). One cut on a
          meaningful feature can really separate the classes. This is why
          trees — and the forests/boosts built on them — **dominate tabular
          leaderboards**.
        - **Images, audio, text** — the raw input is **pixels** (or samples, or
          token ids), and one pixel value means almost nothing by itself. The
          information is in *patterns* — edges, textures, shapes and how they
          sit next to each other. "Pixel $(12, 30) > 0.5$?" is not a meaningful
          question: shift the object by one pixel and the rule is worthless.

        **Why the difference?** A tree can only make **axis-aligned cuts on one
        feature at a time**; it cannot *combine* pixels into an edge detector.
        Recovering a pattern would need an enormous number of nested cuts
        (a deep, overfitting tree). A **neural network** instead *learns* to
        combine pixels into features (edges → parts → objects) — which is why
        **convolutional networks** get the vision jobs and trees get the
        fraud-detection and forecasting jobs.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">9 / 12</div>
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

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">10 / 12</div>
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
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">11 / 12</div>"""),
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

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">12 / 12</div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
