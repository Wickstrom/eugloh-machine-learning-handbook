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
# Lecture 1 (Oct 19), Session 3 — Decision trees.
# Run locally with `marimo edit notebooks/01/03_decision_trees.py`
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
    layout_file="layouts/03_decision_trees.slides.json",
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
        # Decision Trees

        **Machine Learning with Python** — Lecture 1, Session 3 (Oct 19)

        EUGLOH — *Problem Solving Using Open-Source Languages; R and Python*

        University of Novi Sad

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">1 / 22</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Today's session

        - **Motivation** — decisions everywhere; every good question reduces uncertainty
        - **Building blocks** — roots, nodes, branches, leaves
        - **The splitting criterion** — entropy and Gini impurity
        - **CART** — growing a tree greedily; a split computed **by hand**
        - **Interpretability** — reading a tree as rules, and where it breaks down
        - **Practical aspects** — overfitting, pruning, complexity

        Session 3 of 4 today.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">2 / 22</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Part 1 — Decision trees

        ## Motivation

        Our lives are guided (mostly, but not exclusively) by **decisions**:

        - *"Is it raining?"* → take the umbrella or not
        - *"Is the tumor larger than 2 cm?"* → follow-up test or not

        Every question we ask **reduces our uncertainty**; when we reach an
        answer, the uncertainty is zero. Some questions matter more than
        others — a good question resolves a lot of uncertainty at once.

        A decision tree formalises this: **a sequence of questions, asked in
        the order that reduces uncertainty fastest.**

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 22</div>
        """
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.datasets import make_blobs as _make_blobs

    tree_X, tree_y = _make_blobs(
        n_samples=120, centers=[(-1.5, -1.5), (1.5, 1.5)],
        cluster_std=1.0, random_state=3,
    )

    _fig, _ax = _plt.subplots(figsize=(5.5, 4.4))
    _ax.scatter(tree_X[tree_y == 0, 0], tree_X[tree_y == 0, 1],
                color="#dc2626", s=24, label="class 0")
    _ax.scatter(tree_X[tree_y == 1, 0], tree_X[tree_y == 1, 1],
                color="#16a34a", s=24, label="class 1")
    _ax.set_xlabel("$x_1$")
    _ax.set_ylabel("$x_2$")
    _ax.set_title("Which questions would you ask to separate these?")
    _ax.legend()
    _ax.set_aspect("equal")
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.image(_buf, width="620px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 22</div>"""),
        ]
    )
    return tree_X, tree_y


@app.cell
def _(mo, tree_X, tree_y):
    import io as _io

    import matplotlib.pyplot as _plt
    from sklearn.tree import DecisionTreeClassifier as _DTC
    from sklearn.tree import plot_tree as _plot_tree

    _tree = _DTC(max_depth=2, random_state=0).fit(tree_X, tree_y)

    _fig, _ax = _plt.subplots(figsize=(10, 4.4))
    _plot_tree(_tree, feature_names=["$x_1$", "$x_2$"],
               class_names=["class 0", "class 1"], filled=True, rounded=True,
               impurity=False, fontsize=11, ax=_ax)
    for _txt in _ax.texts:  # drop the "samples"/"value" clutter
        _txt.set_text("\n".join(
            _ln for _ln in _txt.get_text().split("\n")
            if not _ln.startswith(("samples", "value"))
        ))
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.md(
                r"""Fitted to **slide 4's data**, here is the tree it grows.
                The first question is the **root**; every box that asks a
                question is a **node**; **branches** carry the yes/no answer;
                and each final box is a **leaf** that predicts its majority
                class. Inference = start at the root, answer the questions,
                read the leaf."""
            ),
            mo.image(_buf, width="880px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 22</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Building blocks

        A tree is a simple and powerful framework to navigate through choices:

        - **Root** — the first question (sees the whole dataset)
        - **Node** — one question about one feature, e.g. $x_1 > 2.3$?
        - **Branch** — the answers (yes / no)
        - **Leaf** — a final region with a prediction (the majority class)
        - **Depth** — the number of questions from root to leaf

        Growing a tree: start at the root → ask a question at a node →
        **pick the best question** → branch out and repeat until termination.

        It can do **classification** (spam / not spam) and **regression**
        (housing prices). Inference is a walk from root to leaf — fully
        **interpretable**: you can read the model as a flowchart.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">6 / 22</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The splitting criterion

        What makes a question *good*? One that **reduces uncertainty** — so
        first we need a way to measure how mixed up a group of samples is.
        Suppose a fraction $p$ of the samples in a node belong to class 1.

        **Gini impurity** (scikit-learn's default, and the one we will use):

        $$G(p) = 2\,p\,(1-p)$$

        **Entropy** (the information-theory version — same idea, a different
        curve):

        $$H(p) = -p \log_2 p - (1-p) \log_2 (1-p)$$

        - both are $0$ when the node is **pure** (all one class)
        - both are **largest** when the node is an even $50/50$ mix ($p = 0.5$)
        - a good split makes the two children *purer* than the parent

        In plain words: one number that answers *"how mixed is this group?"* —
        $0$ means perfectly pure, larger means more mixed. (The $\log_2$ is just
        a logarithm; you never compute it by hand.)

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">7 / 22</div>
        """
    )
    return


@app.cell
def _(mo, tree_X):
    split_slider = mo.ui.slider(
        start=float(tree_X[:, 0].min()), stop=float(tree_X[:, 0].max()),
        step=0.05, value=0.0,
        label="Split threshold on $x_1$", show_value=True, debounce=True,
    )
    mo.md(
        r"""
        ## Interactive demo — how good is a split?

        CART grows a tree by trying many splits and keeping the ones that
        leave the children **purest**. Here is slide 4's data: the slider
        moves a single cut along $x_1$. The right panel evaluates *every*
        candidate cut — the **weighted impurity** of the two children. The
        lowest point is the split CART would pick.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">8 / 22</div>
        """
    )
    return (split_slider,)


@app.cell
def _(mo, split_slider, tree_X, tree_y):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np

    def _gini(labels):
        if len(labels) == 0:
            return 0.0
        _q = _np.mean(labels)
        return 2 * _q * (1 - _q)

    def _entropy(labels):
        if len(labels) == 0:
            return 0.0
        _q = _np.mean(labels)
        if _q in (0.0, 1.0):
            return 0.0
        return -_q * _np.log2(_q) - (1 - _q) * _np.log2(1 - _q)

    _x1 = tree_X[:, 0]
    _ts = _np.linspace(_x1.min(), _x1.max(), 220)
    _wg = _np.empty(_ts.size)
    _we = _np.empty(_ts.size)
    for _i, _t in enumerate(_ts):
        _L, _R = tree_y[_x1 <= _t], tree_y[_x1 > _t]
        _wg[_i] = (len(_L) * _gini(_L) + len(_R) * _gini(_R)) / len(tree_y)
        _we[_i] = (len(_L) * _entropy(_L) + len(_R) * _entropy(_R)) / len(tree_y)

    _tcur = split_slider.value
    _left = tree_y[_x1 <= _tcur]
    _right = tree_y[_x1 > _tcur]
    _gcur = (len(_left) * _gini(_left) + len(_right) * _gini(_right)) / len(tree_y)
    _tbest = _ts[int(_np.argmin(_wg))]

    _fig, _axes = _plt.subplots(1, 2, figsize=(12, 4.6))
    _axes[0].scatter(tree_X[tree_y == 0, 0], tree_X[tree_y == 0, 1],
                     color="#dc2626", s=22, label="class 0")
    _axes[0].scatter(tree_X[tree_y == 1, 0], tree_X[tree_y == 1, 1],
                     color="#16a34a", s=22, label="class 1")
    _axes[0].axvline(_tcur, color="#111827", lw=2.4)
    _axes[0].axvspan(_x1.min() - 0.3, _tcur, color="#111827", alpha=0.05)
    _axes[0].set_title(f"cut: $x_1 \\leq {_tcur:.2f}$")
    _axes[0].set_xlabel("$x_1$")
    _axes[0].set_ylabel("$x_2$")
    _axes[0].legend()
    _axes[0].set_aspect("equal")

    _axes[1].plot(_ts, _wg, color="#2563eb", lw=2, label="weighted Gini")
    _axes[1].plot(_ts, _we, color="#dc2626", lw=2, label="weighted entropy")
    _axes[1].axvline(_tcur, color="gray", ls="--")
    _axes[1].scatter([_tcur], [_gcur], color="#111827", zorder=5)
    _axes[1].scatter([_tbest], [_wg.min()], color="#16a34a", marker="*", s=150,
                     zorder=6, label=f"best: $x_1 \\leq {_tbest:.2f}$")
    _axes[1].set_xlabel("threshold on $x_1$")
    _axes[1].set_ylabel("weighted impurity")
    _axes[1].set_title(f"current split — weighted Gini {_gcur:.3f}")
    _axes[1].legend(fontsize=8)
    _axes[1].grid(alpha=0.3)

    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            split_slider,
            mo.image(_buf, width="920px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">9 / 22</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## CART — growing the tree

        **C**lassification **A**nd **R**egression **T**rees:

        1. Start with **all training samples at the root**
        2. At each node, try every feature and every candidate threshold. For
           each candidate, compute how impure its two children would be, as a
           **weighted average**:

        $$\text{impurity of a split} = \frac{n_L}{n}\,G(\text{left}) + \frac{n_R}{n}\,G(\text{right})$$

        where $n_L$ and $n_R$ are the numbers of samples going left and right.

        3. Pick the split with the **lowest** weighted impurity (greedy!)
        4. Recurse on each child — until a **stopping condition**
           (max depth, pure nodes, min samples per leaf, …)

        The result is a **binary tree**; a leaf predicts its majority class.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">10 / 22</div>
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

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">11 / 22</div>
        """
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np

    _x = _np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    _y = _np.array([0, 0, 0, 1, 0, 1])

    def _gini(labels):
        if len(labels) == 0:
            return 0.0
        _p = _np.mean(labels)
        return 1.0 - _p**2 - (1 - _p) ** 2

    _ts = [1.5, 2.5, 3.5, 4.5, 5.5]
    _rows = []
    _ws = []
    for _t in _ts:
        _L, _R = _y[_x <= _t], _y[_x > _t]
        _gl, _gr = _gini(_L), _gini(_R)
        _w = (len(_L) * _gl + len(_R) * _gr) / len(_y)
        _ws.append(_w)
        _rows.append((_t, _gl, _gr, _w))
    _best = _ts[int(_np.argmin(_ws))]

    _lines = ["| split | G(left) | G(right) | weighted |",
              "|---|---|---|---|"]
    for _t, _gl, _gr, _w in _rows:
        _tag = " ← best" if _t == _best else ""
        _lines.append(
            f"| $x \\leq {_t}$ | {_gl:.3f} | {_gr:.3f} | **{_w:.3f}**{_tag} |"
        )
    _table = "\n".join(_lines)

    _fig, _ax = _plt.subplots(figsize=(7.2, 3.8))
    _colors = ["#16a34a" if _t == _best else "#94a3b8" for _t in _ts]
    _ax.bar([str(_t) for _t in _ts], _ws, color=_colors)
    _ax.set_xlabel("threshold $t$")
    _ax.set_ylabel("weighted Gini")
    _ax.set_title(f"lower is better — best split: $x \\leq {_best}$")
    for _i, _w in enumerate(_ws):
        _ax.text(_i, _w, f"{_w:.3f}", ha="center", va="bottom", fontsize=8)
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.md(_table),
            mo.image(_buf, width="560px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">12 / 22</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    import numpy as _np
    from sklearn.tree import DecisionTreeClassifier as _DTC

    _x = _np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(-1, 1)
    _y = _np.array([0, 0, 0, 1, 0, 1])
    _stump = _DTC(max_depth=1, random_state=0).fit(_x, _y)
    _thr = _stump.tree_.threshold[0]
    mo.md(
        rf"""
        ## The same split, from scikit-learn

        ```python
        from sklearn.tree import DecisionTreeClassifier
        stump = DecisionTreeClassifier(max_depth=1).fit(x, y)
        stump.tree_.threshold[0]   # -> {_thr:.1f}
        ```

        sklearn's root question is **$x \leq {_thr:.1f}$** — exactly the
        threshold our hand calculation selected. The algorithm tried every
        candidate and kept the one with the lowest weighted Gini, just as we
        did.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">13 / 22</div>
        """
    )
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

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">14 / 22</div>
        """
    )
    return


@app.cell
def _(mo, tree_X, tree_y):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.tree import DecisionTreeClassifier as _DTC
    from sklearn.tree import plot_tree as _plot_tree

    _tree = _DTC(max_depth=1, random_state=0).fit(tree_X, tree_y)

    _fig, _axes = _plt.subplots(
        1, 2, figsize=(13.5, 4.6), gridspec_kw={"width_ratios": [1.35, 1]}
    )
    _plot_tree(_tree, feature_names=["$x_1$", "$x_2$"],
               class_names=["class 0", "class 1"], filled=True, rounded=True,
               impurity=False, fontsize=10, ax=_axes[0])
    for _txt in _axes[0].texts:  # drop the "samples"/"value" clutter
        _txt.set_text("\n".join(
            _ln for _ln in _txt.get_text().split("\n")
            if not _ln.startswith(("samples", "value"))
        ))
    _xx, _yy = _np.meshgrid(
        _np.linspace(tree_X[:, 0].min() - 1, tree_X[:, 0].max() + 1, 250),
        _np.linspace(tree_X[:, 1].min() - 1, tree_X[:, 1].max() + 1, 250),
    )
    _Z = _tree.predict(_np.c_[_xx.ravel(), _yy.ravel()]).reshape(_xx.shape)
    _axes[1].contourf(_xx, _yy, _Z, alpha=0.25, cmap="RdYlGn")
    _axes[1].scatter(tree_X[tree_y == 0, 0], tree_X[tree_y == 0, 1],
                     color="#dc2626", s=20, label="class 0")
    _axes[1].scatter(tree_X[tree_y == 1, 0], tree_X[tree_y == 1, 1],
                     color="#16a34a", s=20, label="class 1")
    _axes[1].set_title("one question → one cut")
    _axes[1].set_aspect("equal")
    _axes[1].legend(fontsize=8)
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.md(r"""**Level 1.** The tree asks a *single* question, so it can
            make exactly **one axis-aligned cut** — the plane is split into
            two rectangles."""),
            mo.image(_buf, width="900px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">15 / 22</div>"""),
        ]
    )
    return


@app.cell
def _(mo, tree_X, tree_y):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.tree import DecisionTreeClassifier as _DTC
    from sklearn.tree import plot_tree as _plot_tree

    _tree = _DTC(max_depth=2, random_state=0).fit(tree_X, tree_y)

    _fig, _axes = _plt.subplots(
        1, 2, figsize=(13.5, 4.6), gridspec_kw={"width_ratios": [1.35, 1]}
    )
    _plot_tree(_tree, feature_names=["$x_1$", "$x_2$"],
               class_names=["class 0", "class 1"], filled=True, rounded=True,
               impurity=False, fontsize=10, ax=_axes[0])
    for _txt in _axes[0].texts:  # drop the "samples"/"value" clutter
        _txt.set_text("\n".join(
            _ln for _ln in _txt.get_text().split("\n")
            if not _ln.startswith(("samples", "value"))
        ))
    _xx, _yy = _np.meshgrid(
        _np.linspace(tree_X[:, 0].min() - 1, tree_X[:, 0].max() + 1, 250),
        _np.linspace(tree_X[:, 1].min() - 1, tree_X[:, 1].max() + 1, 250),
    )
    _Z = _tree.predict(_np.c_[_xx.ravel(), _yy.ravel()]).reshape(_xx.shape)
    _axes[1].contourf(_xx, _yy, _Z, alpha=0.25, cmap="RdYlGn")
    _axes[1].scatter(tree_X[tree_y == 0, 0], tree_X[tree_y == 0, 1],
                     color="#dc2626", s=20, label="class 0")
    _axes[1].scatter(tree_X[tree_y == 1, 0], tree_X[tree_y == 1, 1],
                     color="#16a34a", s=20, label="class 1")
    _axes[1].set_title("each leaf asks again → 4 regions")
    _axes[1].set_aspect("equal")
    _axes[1].legend(fontsize=8)
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.md(r"""**Level 2.** Each leaf from level 1 may ask *another*
            question, carving its region further. Depth 2 gives up to four
            rectangles — notice every boundary is still **axis-aligned**."""),
            mo.image(_buf, width="900px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">16 / 22</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.datasets import load_iris as _load_iris
    from sklearn.tree import DecisionTreeClassifier as _DTC
    from sklearn.tree import plot_tree as _plot_tree

    _iris = _load_iris()
    _X = _iris.data[:, 2:4]  # petal length & width
    _y = _iris.target
    _names = ["petal length (cm)", "petal width (cm)"]
    _classes = list(_iris.target_names)

    _tree = _DTC(max_depth=2, random_state=0).fit(_X, _y)

    _fig, _axes = _plt.subplots(1, 2, figsize=(12.5, 4.6))
    _plot_tree(_tree, feature_names=_names, class_names=_classes, filled=True,
               rounded=True, impurity=False, fontsize=11, ax=_axes[0])
    for _txt in _axes[0].texts:  # drop the "samples"/"value" clutter
        _txt.set_text("\n".join(
            _ln for _ln in _txt.get_text().split("\n")
            if not _ln.startswith(("samples", "value"))
        ))
    _xx, _yy = _np.meshgrid(
        _np.linspace(_X[:, 0].min() - 0.3, _X[:, 0].max() + 0.3, 250),
        _np.linspace(_X[:, 1].min() - 0.3, _X[:, 1].max() + 0.3, 250),
    )
    _Z = _tree.predict(_np.c_[_xx.ravel(), _yy.ravel()]).reshape(_xx.shape)
    _axes[1].contourf(_xx, _yy, _Z, alpha=0.25, cmap="RdYlGn")
    for _k in range(len(_classes)):
        _axes[1].scatter(_X[_y == _k, 0], _X[_y == _k, 1], s=22,
                         label=_classes[_k])
    _axes[1].set_xlabel(_names[0])
    _axes[1].set_ylabel(_names[1])
    _axes[1].set_title("depth 2 — three readable rules")
    _axes[1].legend(fontsize=8)
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.md(
                r"""**Interpretability.** On iris (petal length & width) a
                shallow tree reads like a field guide — three questions, no
                code:

                - petal width $\leq 0.80$ → **setosa**
                - $0.80 <$ petal width $\leq 1.75$ → **versicolor**
                - petal width $> 1.75$ → **virginica**"""
            ),
            mo.image(_buf, width="920px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">17 / 22</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.datasets import make_classification as _make_classification
    from sklearn.tree import DecisionTreeClassifier as _DTC

    _X, _y = _make_classification(
        n_samples=200, n_features=2, n_informative=2, n_redundant=0,
        n_clusters_per_class=1, class_sep=1.3, flip_y=0.15, random_state=7,
    )
    _xx, _yy = _np.meshgrid(
        _np.linspace(_X[:, 0].min() - 0.7, _X[:, 0].max() + 0.7, 300),
        _np.linspace(_X[:, 1].min() - 0.7, _X[:, 1].max() + 0.7, 300),
    )
    _fig, _axes = _plt.subplots(1, 3, figsize=(16, 4.8))
    for _ax, _d in zip(_axes, [1, 3, None]):
        _t = _DTC(max_depth=_d, random_state=0).fit(_X, _y)
        _Z = _t.predict(_np.c_[_xx.ravel(), _yy.ravel()]).reshape(_xx.shape)
        _ax.contourf(_xx, _yy, _Z, alpha=0.22, cmap="RdYlGn")
        _ax.scatter(_X[_y == 0, 0], _X[_y == 0, 1], color="#dc2626", s=16,
                    label="class 0")
        _ax.scatter(_X[_y == 1, 0], _X[_y == 1, 1], color="#16a34a", s=16,
                    label="class 1")
        _label = "unlimited" if _d is None else f"depth {_d}"
        _ax.set_title(f"{_label} — {_t.get_n_leaves()} leaves, "
                      f"train {_t.score(_X, _y):.2f}")
        _ax.set_aspect("equal")
        _ax.legend(fontsize=8, loc="lower left")
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    _full = _DTC(random_state=0).fit(_X, _y)
    _single = int(_np.sum(_full.tree_.n_node_samples == 1))
    mo.vstack(
        [
            mo.md(
                rf"""**Where it breaks down.** Same data, deeper trees. The
                unlimited tree reaches **100% training accuracy** with
                {_full.get_n_leaves()} leaves (and {_single} of them hold a
                single sample). Its boundary is full of **thin slivers** — it
                splits the *same* feature again and again at almost identical
                values (e.g. $x_1 \leq -1.35$ followed by $x_1 \leq -1.37$)
                just to fence off noisy points. Those "rules" fit the noise,
                not the signal."""
            ),
            mo.image(_buf, width="980px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">18 / 22</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    depth_slider = mo.ui.slider(
        start=1, stop=15, step=1, value=2,
        label="max_depth", show_value=True, debounce=True,
    )
    noise_switch = mo.ui.switch(
        label="mistake 15% of the training labels", value=False
    )
    mo.md(
        r"""
        ## Practical aspects — the dark side: unlimited trees overfit

        Keep splitting and every training point gets its own rectangle — the
        tree **memorises noise** and will not generalise.

        **Why the boundary can look deceptively stable.** Deeper trees only
        *refine* the partition with ever-smaller, **axis-aligned rectangles**.
        When the true boundary is smooth and the labels are clean, those
        rectangles simply hug the same curve — the picture barely changes and
        test accuracy plateaus. Overfitting needs something to memorise: flip
        the switch to inject **label noise** and watch the test curve peak and
        then fall as the tree fences off individual mistakes.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">19 / 22</div>
        """
    )
    return depth_slider, noise_switch


@app.cell
def _(depth_slider, mo, noise_switch):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.datasets import make_moons as _make_moons
    from sklearn.model_selection import train_test_split as _split
    from sklearn.tree import DecisionTreeClassifier as _DTC

    _X, _y = _make_moons(n_samples=300, noise=0.3, random_state=0)
    _Xtr, _Xte, _ytr, _yte = _split(
        _X, _y, test_size=0.3, random_state=0, stratify=_y
    )

    # Optionally corrupt the training labels so the tree has noise to memorise.
    _rng = _np.random.default_rng(0)
    _ytr_fit = _ytr.copy()
    _flip = 0.15 if noise_switch.value else 0.0
    _mis = _np.zeros(len(_ytr_fit), dtype=bool)
    if _flip:
        _mis = _rng.random(len(_ytr_fit)) < _flip
        _ytr_fit[_mis] = 1 - _ytr_fit[_mis]

    _depth = int(depth_slider.value)
    _clf = _DTC(max_depth=_depth, random_state=0).fit(_Xtr, _ytr_fit)

    _xx, _yy = _np.meshgrid(
        _np.linspace(_X[:, 0].min() - 0.5, _X[:, 0].max() + 0.5, 250),
        _np.linspace(_X[:, 1].min() - 0.5, _X[:, 1].max() + 0.5, 250),
    )
    _Z = _clf.predict(_np.c_[_xx.ravel(), _yy.ravel()]).reshape(_xx.shape)

    # Train/test accuracy as a function of depth (same noisy labels).
    _depths = _np.arange(1, 21)
    _train = []
    _test = []
    for _d in _depths:
        _t = _DTC(max_depth=int(_d), random_state=0).fit(_Xtr, _ytr_fit)
        _train.append(_t.score(_Xtr, _ytr_fit))
        _test.append(_t.score(_Xte, _yte))

    _fig, _axes = _plt.subplots(1, 2, figsize=(12.5, 4.8))
    _axes[0].contourf(_xx, _yy, _Z, alpha=0.25, cmap="RdYlGn")
    _axes[0].scatter(_Xtr[_ytr_fit == 0, 0], _Xtr[_ytr_fit == 0, 1],
                     color="#dc2626", s=16, label="class 0 (train)")
    _axes[0].scatter(_Xtr[_ytr_fit == 1, 0], _Xtr[_ytr_fit == 1, 1],
                     color="#16a34a", s=16, label="class 1 (train)")
    if _mis.any():
        _axes[0].scatter(_Xtr[_mis, 0], _Xtr[_mis, 1], facecolors="none",
                         edgecolors="k", s=70, label="mislabeled")
    _axes[0].set_title(
        f"max_depth = {_depth} · train {_clf.score(_Xtr, _ytr_fit):.2f} · "
        f"test {_clf.score(_Xte, _yte):.2f}"
    )
    _axes[0].set_aspect("equal")
    _axes[0].legend(fontsize=7, loc="lower left")

    _axes[1].plot(_depths, _train, "o-", color="#2563eb", label="train")
    _axes[1].plot(_depths, _test, "s-", color="#f59e0b", label="test")
    _axes[1].axvline(_depth, color="gray", ls="--")
    _axes[1].set_xlabel("max_depth")
    _axes[1].set_ylabel("accuracy")
    _axes[1].set_ylim(0.5, 1.02)
    _axes[1].set_title(f"label noise = {_flip:.0%}")
    _axes[1].legend()
    _axes[1].grid(alpha=0.3)

    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            depth_slider,
            noise_switch,
            mo.image(_buf, width="920px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">20 / 22</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Practical aspects — keeping trees honest

        | ✅ Advantages | ❌ Disadvantages |
        |---|---|
        | Minimal preprocessing needed | Sensitive to small feature changes |
        | Great for most data types | **Greedy** — no global optimum |
        | Interpretable (a flowchart) | **Overfits** when grown deep |
        | Fast prediction | Training cost $O(m \cdot n \log n)$ — grows gently with data size |

        **Controlling growth:**
        - Limit it — `max_depth`, `min_samples_leaf`, `min_samples_split`
        - **Cost-complexity pruning** — grow a full tree, then gradually
          remove the subtrees ("weakest links") whose removal increases the
          error least per leaf removed; use **cross-validation** to pick the
          best trade-off (`sklearn`: `cost_complexity_pruning_path`, `ccp_alpha`)

        *(For regression trees the recipe is identical — split by variance
        reduction instead of Gini, and leaves predict the mean.)*

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">21 / 22</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Summary

        - **Decision trees** formalise a sequence of uncertainty-reducing
          questions; roots, nodes, branches, leaves.
        - A **good split** minimises **Gini impurity** (or entropy) — computed
          greedily over every feature and threshold (**CART**).
        - Trees are **interpretable** and need little preprocessing — a shallow
          tree reads as rules — but grown deep they invent **conflicting
          splits** and **overfit**; control depth, or **prune**.

        Next session: **random forests**.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">22 / 22</div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
