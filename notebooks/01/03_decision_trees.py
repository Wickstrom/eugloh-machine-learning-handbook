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

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">1 / 18</div>
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
        - **Practical aspects** — overfitting, pruning, complexity

        Session 3 of 4 today.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">2 / 18</div>
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

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 18</div>
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
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 18</div>"""),
        ]
    )
    return tree_X, tree_y


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

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 18</div>
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

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">6 / 18</div>
        """
    )
    return


@app.cell
def _(mo):
    p_slider = mo.ui.slider(
        start=0.01, stop=0.99, step=0.01, value=0.5,
        label="Fraction of class 1 (p)", show_value=True, debounce=True,
    )
    mo.md(
        r"""
        ## Interactive demo — impurity curves

        Move the slider to change the class proportion $p$ in a node, and
        watch both impurity measures react. Where is the impurity maximal?

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">7 / 18</div>
        """
    )
    return (p_slider,)


@app.cell
def _(mo, p_slider):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np

    _p = p_slider.value
    _ps = _np.linspace(0.001, 0.999, 400)
    _entropy = -_ps * _np.log2(_ps) - (1 - _ps) * _np.log2(1 - _ps)
    _gini = 2 * _ps * (1 - _ps)
    _h = -_p * _np.log2(_p) - (1 - _p) * _np.log2(1 - _p)
    _g = 2 * _p * (1 - _p)

    _fig, _ax = _plt.subplots(figsize=(6.6, 4))
    _ax.plot(_ps, _entropy, color="#2563eb", lw=2, label="entropy")
    _ax.plot(_ps, _gini, color="#dc2626", lw=2, label="Gini")
    _ax.axvline(_p, color="gray", ls="--", lw=1)
    _ax.scatter([_p, _p], [_h, _g], color=["#2563eb", "#dc2626"], zorder=5)
    _ax.set_xlabel("p")
    _ax.set_title(f"p = {_p:.2f}  →  entropy = {_h:.3f},  Gini = {_g:.3f}")
    _ax.legend()
    _ax.grid(alpha=0.3)
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            p_slider,
            mo.image(_buf, width="620px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">8 / 18</div>"""),
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

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">9 / 18</div>
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

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">10 / 18</div>
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
        _star = "  <- best" if _t == 3.5 else ""
        print(f"   x <= {_t}  |   {_gini(_L):.3f}     {_gini(_R):.3f}   |  {_w:.3f}{_star}")
    mo.md(
        r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">11 / 18</div>"""
    )
    return


@app.cell
def _(mo):
    import numpy as _np
    from sklearn.tree import DecisionTreeClassifier as _DTC

    _x = _np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(-1, 1)
    _y = _np.array([0, 0, 0, 1, 0, 1])
    _stump = _DTC(max_depth=1, random_state=0).fit(_x, _y)
    print(f"sklearn's root split: x <= {_stump.tree_.threshold[0]:.1f}")
    print("(agrees with the hand calculation — threshold 3.5)")
    mo.md(
        r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">12 / 18</div>"""
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

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">13 / 18</div>
        """
    )
    return


@app.cell
def _(mo, tree_X, tree_y):
    import io as _io

    import matplotlib.pyplot as _plt
    from sklearn.tree import DecisionTreeClassifier as _DTC
    from sklearn.tree import plot_tree as _plot_tree

    _tree = _DTC(max_depth=2, random_state=0).fit(tree_X, tree_y)

    _fig1, _ax = _plt.subplots(figsize=(10, 4.2))
    _plot_tree(_tree, feature_names=["$x_1$", "$x_2$"],
               class_names=["class 0", "class 1"], filled=True, ax=_ax)
    _buf1 = _io.BytesIO()
    _fig1.savefig(_buf1, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig1)
    _buf1.seek(0)

    import numpy as _np
    _xx, _yy = _np.meshgrid(
        _np.linspace(tree_X[:, 0].min() - 1, tree_X[:, 0].max() + 1, 250),
        _np.linspace(tree_X[:, 1].min() - 1, tree_X[:, 1].max() + 1, 250),
    )
    _Z = _tree.predict(_np.c_[_xx.ravel(), _yy.ravel()]).reshape(_xx.shape)

    _fig2, _ax2 = _plt.subplots(figsize=(5.2, 4.2))
    _ax2.contourf(_xx, _yy, _Z, alpha=0.25, cmap="RdYlGn")
    _ax2.scatter(tree_X[tree_y == 0, 0], tree_X[tree_y == 0, 1], color="#dc2626", s=20)
    _ax2.scatter(tree_X[tree_y == 1, 0], tree_X[tree_y == 1, 1], color="#16a34a", s=20)
    _ax2.set_title("Depth-2 tree — axis-aligned cuts")
    _ax2.set_aspect("equal")
    _buf2 = _io.BytesIO()
    _fig2.savefig(_buf2, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig2)
    _buf2.seek(0)
    mo.vstack(
        [
            mo.image(_buf1, width="860px"),
            mo.image(_buf2, width="520px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">14 / 18</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    depth_slider = mo.ui.slider(
        start=1, stop=15, step=1, value=2,
        label="max_depth", show_value=True, debounce=True,
    )
    mo.md(
        r"""
        ## Practical aspects — the dark side: unlimited trees overfit

        Keep splitting and every training point gets its own rectangle — the
        tree **memorises noise** and will not generalise.

        Watch the boundary below as `max_depth` grows: train accuracy climbs
        towards 1.0, but **test** accuracy peaks early and then decays.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">15 / 18</div>
        """
    )
    return (depth_slider,)


@app.cell
def _(depth_slider, mo):
    import io as _io

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

    _fig, _ax = _plt.subplots(figsize=(5.6, 4.6))
    _ax.contourf(_xx, _yy, _Z, alpha=0.25, cmap="RdYlGn")
    _ax.scatter(_X[_y == 0, 0], _X[_y == 0, 1], color="#dc2626", s=16)
    _ax.scatter(_X[_y == 1, 0], _X[_y == 1, 1], color="#16a34a", s=16)
    _ax.set_title(
        f"max_depth = {depth_slider.value} · "
        f"train {_clf.score(_Xtr, _ytr):.2f} · test {_clf.score(_Xte, _yte):.2f}"
    )
    _ax.set_aspect("equal")
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            depth_slider,
            mo.image(_buf, width="620px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">16 / 18</div>"""),
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

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">17 / 18</div>
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
        - Trees are **interpretable** and need little preprocessing, but they
          **overfit** if grown deep — control depth, or **prune**.

        Next session: **random forests**.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">18 / 18</div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
