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
# Lecture 2 — Decision Trees and Random Forests.
# Run locally with `marimo edit notebooks/02/decision_trees_random_forests.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).

import marimo

__generated_with = "0.17.6"
app = marimo.App(
    width="medium",
    layout_file="layouts/decision_trees_random_forests.slides.json",
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
        # Decision Trees and Random Forests

        **Machine Learning with Python** — Lecture 2

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

        - **Decision trees** — classifiers built from simple yes/no questions
        - How to find the *best* question: **impurity** (Gini, entropy)
        - Why unlimited trees **overfit** — and how to stop them
        - **Random forests** — combining many trees into a much stronger one

        TODO: recap last lecture in one sentence before starting.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Part 1 — Decision trees

        ## Motivation

        Our lives are guided by decisions — and every good question reduces
        our **uncertainty** about the answer:

        - *"Is it raining?"* → take the umbrella or not
        - *"Is the tumor larger than 2 cm?"* → follow-up test or not

        A decision tree formalises this: **a sequence of questions, asked in
        the order that reduces uncertainty fastest.**

        Logistic regression draws one straight line. A tree cuts the feature
        space with **axis-aligned** rules — and can approximate much more
        complex boundaries.
        """
    )
    return


@app.cell
def _(mo):
    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.datasets import make_blobs as _make_blobs

    tree_X, tree_y = _make_blobs(
        n_samples=120, centers=[(-1.5, -1.5), (1.5, 1.5)],
        cluster_std=1.0, random_state=3,
    )

    _fig, _ax = _plt.subplots(figsize=(5.5, 4.5))
    _ax.scatter(tree_X[tree_y == 0, 0], tree_X[tree_y == 0, 1],
                color="#dc2626", s=24, label="class 0")
    _ax.scatter(tree_X[tree_y == 1, 0], tree_X[tree_y == 1, 1],
                color="#16a34a", s=24, label="class 1")
    _ax.set_xlabel("$x_1$")
    _ax.set_ylabel("$x_2$")
    _ax.set_title("Which questions would you ask to separate these?")
    _ax.legend()
    _ax.set_aspect("equal")
    mo.as_html(_fig)
    _plt.close(_fig)
    return tree_X, tree_y


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Terminology

        - **Root** — the first question (the whole dataset)
        - **Node** — one question about one feature, e.g. $x_1 > 2.3$?
        - **Branch** — the answers (yes / no)
        - **Leaf** — a final region with a prediction (the majority class)
        - **Depth** — the number of questions from root to leaf

        Inference is a walk from root to leaf — fully **interpretable**:
        you can read the model as a flowchart.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Which question is "best"? Impurity

        We measure how *mixed* the classes in a node are. For a node where a
        fraction $p$ of samples belong to class 1:

        **Entropy**

        $$H(p) = -p \log_2 p - (1-p) \log_2 (1-p)$$

        **Gini impurity** (sklearn's default)

        $$G(p) = 2\,p\,(1-p)$$

        - Both are $0$ for a **pure** node (all one class) and maximal at $p = 0.5$
        - A good split makes the children *purer* than the parent
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Interactive demo — impurity curves

        Move the slider to change the class proportion $p$ in a node, and
        watch both impurity measures react.
        """
    )
    return


@app.cell
def _(mo):
    p_slider = mo.ui.slider(
        start=0.01, stop=0.99, step=0.01, value=0.5,
        label="Fraction of class 1 (p)", show_value=True,
    )
    p_slider
    return (p_slider,)


@app.cell
def _(mo, p_slider):
    import matplotlib.pyplot as _plt
    import numpy as _np

    _p = p_slider.value
    _ps = _np.linspace(0.001, 0.999, 400)
    _entropy = -_ps * _np.log2(_ps) - (1 - _ps) * _np.log2(1 - _ps)
    _gini = 2 * _ps * (1 - _ps)
    _h = -_p * _np.log2(_p) - (1 - _p) * _np.log2(1 - _p)
    _g = 2 * _p * (1 - _p)

    _fig, _ax = _plt.subplots(figsize=(7, 4))
    _ax.plot(_ps, _entropy, color="#2563eb", lw=2, label="entropy")
    _ax.plot(_ps, _gini, color="#dc2626", lw=2, label="Gini")
    _ax.axvline(_p, color="gray", ls="--", lw=1)
    _ax.scatter([_p, _p], [_h, _g], color=["#2563eb", "#dc2626"], zorder=5)
    _ax.set_xlabel("p")
    _ax.set_title(f"p = {_p:.2f}  →  entropy = {_h:.3f},  Gini = {_g:.3f}")
    _ax.legend()
    _ax.grid(alpha=0.3)
    mo.as_html(_fig)
    _plt.close(_fig)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The CART algorithm

        **C**lassification **A**nd **R**egression **T**rees:

        1. Start with all training samples at the root
        2. For every feature and every candidate threshold, compute the
           **weighted impurity** of the two children:

        $$G(\text{split at } t) = \frac{n_L}{n} G(\text{left}) + \frac{n_R}{n} G(\text{right})$$

        3. Pick the split with the **lowest** weighted impurity (greedy!)
        4. Recurse on each child — until a stopping condition
           (max depth, pure nodes, min samples per leaf, …)

        The result is a **binary tree**; a leaf predicts its majority class.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Worked example — one split, by hand

        Six samples, one feature:

        | $x$ | 1 | 2 | 3 | 4 | 5 | 6 |
        |---|---|---|---|---|---|---|
        | $y$ | 0 | 0 | 0 | 1 | 0 | 1 |

        Candidate thresholds sit between points: $t \in \{1.5, 2.5, 3.5, 4.5, 5.5\}$.

        Try it with pen and paper first — then let the code check you.
        """
    )
    return


@app.cell
def _(mo):
    import numpy as _np

    _x = _np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    _y = _np.array([0, 0, 0, 1, 0, 1])

    def _gini(labels):
        if len(labels) == 0:
            return 0.0
        p = _np.mean(labels)
        return 1.0 - p**2 - (1 - p) ** 2

    print("threshold |  G(left)  G(right) | weighted")
    print("----------+--------------------+---------")
    for _t in [1.5, 2.5, 3.5, 4.5, 5.5]:
        _L, _R = _y[_x <= _t], _y[_x > _t]
        _w = (len(_L) * _gini(_L) + len(_R) * _gini(_R)) / len(_y)
        _star = "  ← best" if _t == 3.5 else ""
        print(f"   x ≤ {_t}  |   {_gini(_L):.3f}     {_gini(_R):.3f}   |  {_w:.3f}{_star}")
    return


@app.cell
def _(mo):
    import numpy as _np
    from sklearn.tree import DecisionTreeClassifier as _DTC

    _x = _np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(-1, 1)
    _y = _np.array([0, 0, 0, 1, 0, 1])
    _stump = _DTC(max_depth=1, random_state=0).fit(_x, _y)
    print(f"sklearn's root split: x ≤ {_stump.tree_.threshold[0]:.1f}")
    print("(agrees with the hand calculation — threshold 3.5)")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Decision trees in scikit-learn

        ```python
        from sklearn.tree import DecisionTreeClassifier, plot_tree
        tree = DecisionTreeClassifier(max_depth=2).fit(X, y)
        plot_tree(tree, feature_names=[...], class_names=[...], filled=True)
        ```

        Fitting is instant on small data — and the tree can be *drawn*.
        """
    )
    return


@app.cell
def _(mo, tree_X, tree_y):
    import matplotlib.pyplot as _plt
    from sklearn.tree import DecisionTreeClassifier as _DTC
    from sklearn.tree import plot_tree as _plot_tree

    _tree = _DTC(max_depth=2, random_state=0).fit(tree_X, tree_y)

    _fig1, _ax = _plt.subplots(figsize=(10, 4.5))
    _plot_tree(_tree, feature_names=["$x_1$", "$x_2$"],
               class_names=["class 0", "class 1"], filled=True, ax=_ax)
    mo.as_html(_fig1)
    _plt.close(_fig1)

    import numpy as _np
    _xx, _yy = _np.meshgrid(
        _np.linspace(tree_X[:, 0].min() - 1, tree_X[:, 0].max() + 1, 250),
        _np.linspace(tree_X[:, 1].min() - 1, tree_X[:, 1].max() + 1, 250),
    )
    _Z = _tree.predict(_np.c_[_xx.ravel(), _yy.ravel()]).reshape(_xx.shape)

    _fig2, _ax2 = _plt.subplots(figsize=(5.5, 4.5))
    _ax2.contourf(_xx, _yy, _Z, alpha=0.25, cmap="RdYlGn")
    _ax2.scatter(tree_X[tree_y == 0, 0], tree_X[tree_y == 0, 1], color="#dc2626", s=20)
    _ax2.scatter(tree_X[tree_y == 1, 0], tree_X[tree_y == 1, 1], color="#16a34a", s=20)
    _ax2.set_title("Depth-2 tree — axis-aligned cuts")
    _ax2.set_aspect("equal")
    mo.as_html(_fig2)
    _plt.close(_fig2)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The dark side: unlimited trees overfit

        Keep splitting and every training point gets its own rectangle — the
        tree memorises noise and will not generalise.

        Watch the boundary below as `max_depth` grows: train accuracy climbs
        towards 1.0, but **test** accuracy peaks early and then decays.
        """
    )
    return


@app.cell
def _(mo):
    depth_slider = mo.ui.slider(
        start=1, stop=15, step=1, value=2,
        label="max_depth", show_value=True,
    )
    depth_slider
    return (depth_slider,)


@app.cell
def _(depth_slider, mo):
    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.datasets import make_moons as _make_moons
    from sklearn.model_selection import train_test_split as _split
    from sklearn.tree import DecisionTreeClassifier as _DTC

    _X, _y = _make_moons(n_samples=300, noise=0.3, random_state=0)
    _Xtr, _Xte, _ytr, _yte = _split(_X, _y, test_size=0.3, random_state=0, stratify=_y)
    _clf = _DTC(max_depth=depth_slider.value, random_state=0).fit(_Xtr, _ytr)

    _xx, _yy = _np.meshgrid(
        _np.linspace(_X[:, 0].min() - 0.5, _X[:, 0].max() + 0.5, 250),
        _np.linspace(_X[:, 1].min() - 0.5, _X[:, 1].max() + 0.5, 250),
    )
    _Z = _clf.predict(_np.c_[_xx.ravel(), _yy.ravel()]).reshape(_xx.shape)

    _fig, _ax = _plt.subplots(figsize=(6, 5))
    _ax.contourf(_xx, _yy, _Z, alpha=0.25, cmap="RdYlGn")
    _ax.scatter(_X[_y == 0, 0], _X[_y == 0, 1], color="#dc2626", s=16)
    _ax.scatter(_X[_y == 1, 0], _X[_y == 1, 1], color="#16a34a", s=16)
    _ax.set_title(
        f"max_depth = {depth_slider.value} · "
        f"train {_clf.score(_Xtr, _ytr):.2f} · test {_clf.score(_Xte, _yte):.2f}"
    )
    _ax.set_aspect("equal")
    mo.as_html(_fig)
    _plt.close(_fig)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Keeping trees honest

        - **Limit growth** — `max_depth`, `min_samples_leaf`, `min_samples_split`
        - **Cost-complexity pruning** — grow a full tree, then remove the
          subtrees whose removal hurts accuracy the least
          (`sklearn`: `cost_complexity_pruning_path` + cross-validation over `ccp_alpha`)
        - Trees are **greedy** — no guarantee of the globally best tree,
          but fast and usually effective

        *(For regression trees the recipe is identical — just split by
        variance reduction instead of Gini, and leaves predict the mean.)*
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Part 2 — Random forests

        ## The power of the crowd

        A single deep tree overfits; a single shallow tree underfits.

        **Ensemble** many *weak* trees into one *strong* learner:

        1. **Bagging** — train each tree on a random bootstrap sample of the data
        2. **Random feature subsets** — at each split, only consider a random
           subset of features → trees become *different* from each other
        3. **Aggregate** — majority vote (classification) or average (regression)

        Averaging many decorrelated trees cancels their individual errors.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Interactive demo — grow a forest

        More trees smooth the boundary; unlimited depth lets each tree
        overfit. Find the sweet spot!
        """
    )
    return


@app.cell
def _(mo):
    n_trees_slider = mo.ui.slider(
        start=1, stop=200, step=1, value=50,
        label="n_estimators", show_value=True,
    )
    forest_depth_slider = mo.ui.slider(
        start=1, stop=15, step=1, value=5,
        label="max_depth per tree", show_value=True,
    )
    mo.hstack([n_trees_slider, forest_depth_slider])
    return forest_depth_slider, n_trees_slider


@app.cell
def _(forest_depth_slider, mo, n_trees_slider):
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

    _fig, _ax = _plt.subplots(figsize=(6, 5))
    _ax.contourf(_xx, _yy, _Z, alpha=0.25, cmap="RdYlGn")
    _ax.scatter(_X[_y == 0, 0], _X[_y == 0, 1], color="#dc2626", s=16)
    _ax.scatter(_X[_y == 1, 0], _X[_y == 1, 1], color="#16a34a", s=16)
    _ax.set_title(
        f"{n_trees_slider.value} trees, depth {forest_depth_slider.value} · "
        f"train {_clf.score(_Xtr, _ytr):.2f} · test {_clf.score(_Xte, _yte):.2f}"
    )
    _ax.set_aspect("equal")
    mo.as_html(_fig)
    _plt.close(_fig)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## What did the forest learn? Feature importance

        sklearn averages how much each feature decreases impurity across all
        trees — a rough but useful measure of which features matter.

        TODO: mention permutation importance as a more reliable alternative.
        """
    )
    return


@app.cell
def _(mo):
    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.datasets import load_wine as _load_wine
    from sklearn.ensemble import RandomForestClassifier as _RFC

    _wine = _load_wine()
    _forest = _RFC(n_estimators=200, random_state=0).fit(_wine.data, _wine.target)
    _imp = _forest.feature_importances_
    _order = _np.argsort(_imp)

    _fig, _ax = _plt.subplots(figsize=(8, 6))
    _ax.barh(_np.array(_wine.feature_names)[_order], _imp[_order], color="#2563eb")
    _ax.set_title("Random-forest feature importance (wine dataset)")
    _ax.set_xlabel("importance")
    _fig.tight_layout()
    mo.as_html(_fig)
    _plt.close(_fig)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Trees & forests — strengths and weaknesses

        | | |
        |---|---|
        | ✅ No scaling needed, handles mixed feature types | ❌ Single trees overfit easily |
        | ✅ Interpretable (a single tree is a flowchart) | ❌ Forests lose the interpretability |
        | ✅ Non-linear, capture interactions automatically | ❌ Boundaries are axis-aligned (until you ensemble) |
        | ✅ Excellent on tabular data — often beat neural networks | ❌ Large forests = more memory, slower inference |

        Most real-world data lives in tables — this is why
        trees/boosting/forests dominate Kaggle leaderboards.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Summary

        - A decision tree asks greedy yes/no questions, chosen to minimise
          **weighted Gini/entropy** — you computed one split **by hand** and
          scikit-learn agreed exactly.
        - Unlimited depth **overfits**; control it with `max_depth`,
          `min_samples_leaf`, or cost-complexity pruning.
        - **Random forests** = bagging + random feature subsets: many
          decorrelated trees vote, errors cancel, accuracy climbs.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Where to go next

        - **Exercise:** `notebooks/02/decision_trees_random_forests_exercise.ipynb`
          — implement Gini impurity and best-split search by hand, verify
          against sklearn, then benchmark trees vs forests on wine & breast cancer.
        - **Lecture 3:** neural networks — smooth, fully non-linear boundaries.
        """
    )
    return


if __name__ == "__main__":
    app.run()
