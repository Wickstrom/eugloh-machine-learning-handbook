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
# On-site (Oct 23-24) — Project introduction lecture (30 minutes).
# Sets up the hands-on project: what we will build, the aurora dataset, the
# paper we follow (Kvammen et al., 2020), and the two ideas the whole project
# rests on — dimensionality reduction (PCA / t-SNE) and pretrained features.
#
# Run locally with `marimo edit notebooks/03/00_aurora_project.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).
#
# NOTE on scoping: Marimo requires each global name to be owned by exactly one
# cell, so all cell-locals are underscore-prefixed. Figures are drawn with
# matplotlib and shown as images, so the deck runs unchanged in the browser.

import marimo

__generated_with = "0.17.6"
app = marimo.App(
    width="medium",
    layout_file="layouts/00_aurora_project.slides.json",
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
        # The Aurora Project

        ## Classifying the northern lights

        **Machine Learning with Python** — On-site project (Oct 23–24)

        EUGLOH — *Problem Solving Using Open-Source Languages; R and Python*

        University of Novi Sad

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">1 / 13</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## What we will do

        Over the next two days you will build a small machine-learning pipeline
        that looks at a photo of the sky and says **what kind of aurora** is in it.

        - **You already have all the tools** — plotting, PCA/t-SNE, logistic
          regression, trees, forests, a small neural network.
        - **You will work through a real research dataset** — 3846 images of the
          sky above Kiruna, Sweden, each labelled with one of seven auroral types.
        - **You will not train a big network from scratch.** We reuse a
          **pretrained** network as a feature extractor — the same trick used in
          the paper this project is based on.

        **The project comes in five levels**, from *Beginner* to *God-like*.
        Pick the one that matches your comfort — they all use the same data.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">2 / 13</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The data — seven kinds of aurora

        Images from the **all-sky camera in Kiruna** (Swedish Institute of Space
        Physics), 2010–2019, down-sampled to $128\times128$ pixels.

        | label | class | label | class |
        |:-----:|:------|:-----:|:------|
        | 0 | Arc | 4 | Edge |
        | 1 | Auroral breakup | 5 | Faint |
        | 2 | Colored | 6 | Patchy |
        | 3 | Discrete | | |

        - **3000 training** + **846 test** images (the test set is the one we
          report on — never train on it).
        - The classes are **imbalanced**: *Patchy* is by far the largest, *Faint*
          is small and looks almost black.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 13</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.image(
                str(mo.notebook_location() / "public" / "aurora_class_samples.png"),
                width="720px",
            ),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 13</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The paper we follow

        **Kvammen, Wickstrøm, McKay & Partamies (2020)**,
        *Auroral Image Classification With Deep Neural Networks*,
        JGR: Space Physics. [doi:10.1029/2020JA027808](https://doi.org/10.1029/2020JA027808)

        - Introduces these **seven auroral subclasses** and the labelled dataset.
        - Compares deep networks against classic methods (**k-NN, SVM**).
        - **Deep networks won**; **ResNet-50 reached ~92% precision**.

        Its **Figure 5** is the picture we will rebuild: every test image is
        pushed through the network, the network's **features are projected to two
        dimensions with t-SNE**, and a few example images are drawn on the map.

        The good news: **we have already computed the features for you**, so the
        entry level only has to plot them.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 13</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.image(
                str(mo.notebook_location() / "public" / "aurora_paper_fig5.jpg"),
                width="780px",
            ),
            mo.md(
                r"""*Kvammen et al. (2020), Figure 5 — reproduced under CC BY. The ResNet-50 features of the test set, projected to 2-D with t-SNE, with example images drawn on the map. **This is the figure we rebuild in Level 1** (we use ResNet-18).*

                <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">6 / 13</div>"""
            ),
        ]
    )
    return


@app.cell
def _(mo):
    import io as _io
    import matplotlib.pyplot as _plt
    from matplotlib.patches import FancyBboxPatch as _Box, FancyArrowPatch as _Arrow

    _ink = "#0f172a"
    _pal = {
        "image": ("#eef2ff", "#6366f1"),
        "cnn": ("#e0f2fe", "#0284c7"),
        "feat": ("#dcfce7", "#16a34a"),
        "reduce": ("#ede9fe", "#7c3aed"),
        "clf": ("#fef3c7", "#d97706"),
    }

    def _bx(ax, xy, wh, text, kind, fs=11):
        x, y = xy
        w, h = wh
        fc, ec = _pal[kind]
        ax.add_patch(_Box((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.14",
                          lw=1.8, edgecolor=ec, facecolor=fc, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=_ink, zorder=3)

    def _ar(ax, p, q):
        ax.add_patch(_Arrow(p, q, arrowstyle="-|>", mutation_scale=18, lw=2.0,
                            color="#475569", shrinkA=2, shrinkB=2, zorder=1))

    _fig, _ax = _plt.subplots(figsize=(12, 3.6))
    _ax.set_xlim(0, 12)
    _ax.set_ylim(0, 3.6)
    _ax.set_aspect("equal")
    _ax.axis("off")

    _bx(_ax, (0.15, 1.35), (1.9, 0.9), "raw image", "image")
    _bx(_ax, (2.65, 1.25), (2.7, 1.1), "pretrained CNN\n(frozen)", "cnn")
    _bx(_ax, (5.95, 1.35), (2.1, 0.9), "feature vector\n(512-d)", "feat")
    _bx(_ax, (8.75, 2.05), (3.05, 0.8), "reduce to 2-D (PCA / t-SNE)\nto look", "reduce", fs=10)
    _bx(_ax, (8.75, 0.45), (3.05, 0.8), "classifier\nto predict the class", "clf", fs=10)
    _ar(_ax, (2.05, 1.8), (2.65, 1.8))
    _ar(_ax, (5.35, 1.8), (5.95, 1.8))
    _ar(_ax, (8.05, 2.0), (8.75, 2.45))
    _ar(_ax, (8.05, 1.6), (8.75, 0.85))

    _ax.set_title("The whole project in one picture", fontsize=13, color=_ink, pad=10)
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.image(_buf, width="900px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">7 / 13</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    import io as _io
    import matplotlib.pyplot as _plt
    from matplotlib.patches import (
        FancyBboxPatch as _FBP,
        FancyArrowPatch as _FAP,
        Rectangle as _Rect,
    )

    _ink = "#0f172a"
    _muted = "#475569"

    _fig, _ax = _plt.subplots(figsize=(11.5, 3.4))
    _ax.set_xlim(0, 11.5)
    _ax.set_ylim(0, 3.4)
    _ax.set_aspect("equal")
    _ax.axis("off")

    _ax.add_patch(_Rect((0.2, 0.9), 1.5, 1.5, facecolor="#0b3d1f",
                        edgecolor="#334155", lw=1.5, zorder=2))
    _ax.text(0.95, 0.45, "128$\\times$128 image", ha="center", fontsize=9, color=_muted)

    for _x, _s in zip([2.6, 4.6, 6.6], [1.5, 1.1, 0.8]):
        for _k in (2, 1, 0):
            _d = _k * 0.12
            _ax.add_patch(_FBP((_x + _d, 0.9 + _d), _s, _s,
                               boxstyle="round,pad=0.0,rounding_size=0.06",
                               facecolor="#bae6fd", edgecolor="#0284c7",
                               lw=1.2, alpha=0.9, zorder=2 + _k))
    _ax.text(5.3, 0.45, "convolutional feature maps", ha="center", fontsize=9, color=_muted)

    _fx = 8.4
    for _i in range(10):
        _ax.add_patch(_Rect((_fx, 0.7 + _i * 0.18), 0.45, 0.16,
                            facecolor="#bbf7d0", edgecolor="#16a34a", lw=1.0, zorder=2))
    _ax.text(_fx + 0.22, 0.45, "feature vector (512-d)", ha="center", fontsize=9, color=_muted)

    for _p, _q in [((1.75, 1.65), (2.55, 1.65)), ((4.25, 1.65), (4.55, 1.65)),
                   ((6.25, 1.65), (6.55, 1.65)), ((7.6, 1.65), (8.35, 1.65))]:
        _ax.add_patch(_FAP(_p, _q, arrowstyle="-|>", mutation_scale=18, lw=2.0,
                           color=_muted, shrinkA=2, shrinkB=2, zorder=1))

    _ax.text(9.95, 2.5, "similar images\nland near each other", ha="center", va="center",
             fontsize=10, color=_ink,
             bbox=dict(boxstyle="round,pad=0.4", fc="#f8fafc", ec="#cbd5e1"))
    _ax.annotate("", xy=(8.95, 1.9), xytext=(9.7, 2.25),
                 arrowprops=dict(arrowstyle="-|>", color=_muted, lw=1.4))
    _ax.set_title("From an image to a feature vector", fontsize=13, color=_ink, pad=10)

    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.md(
                r"""## Idea 1 — pretrained features

Training an image model from scratch needs millions of labelled images and many
GPUs. Instead we take a network already trained on a huge dataset (ImageNet) and
**keep the layer just before its classifier**.

- For any image, that layer produces a **feature vector** — an **embedding**
  (512 numbers).
- **Similar images land near each other** in this space.
- We then feed the embedding to **any** classifier we know — logistic regression,
  a tree, a forest, a small MLP.

**In plain words: turn each image into a compact vector, then use the models from
earlier in the course.**"""
            ),
            mo.image(_buf, width="820px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">8 / 13</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    import io as _io
    import matplotlib.pyplot as _plt
    from sklearn.datasets import load_iris as _load_iris
    from sklearn.decomposition import PCA as _PCA
    from sklearn.preprocessing import StandardScaler as _StandardScaler

    _ink = "#0f172a"
    _iris = _load_iris()
    _X, _y = _iris.data, _iris.target
    _cols = ["#ef4444", "#3b82f6", "#22c55e"]
    _species = ["setosa", "versicolor", "virginica"]

    _fig = _plt.figure(figsize=(11, 3.6), layout="constrained")

    # --- left: we can only draw three of the four features ---
    _ax1 = _fig.add_subplot(1, 2, 1, projection="3d")
    for _i, (_c, _s) in enumerate(zip(_cols, _species)):
        _m = _y == _i
        _ax1.scatter(_X[_m, 0], _X[_m, 1], _X[_m, 2], s=22, color=_c,
                     alpha=0.85, label=_s, depthshade=False)
    _ax1.set_xlabel("feature 1  (sepal len)", fontsize=8)
    _ax1.set_ylabel("feature 2  (sepal wid)", fontsize=8)
    _ax1.set_zlabel("feature 3  (petal len)", fontsize=8)
    _ax1.set_title("4 features — we can only draw 3", fontsize=11, color=_ink)
    _ax1.tick_params(labelsize=7)
    _ax1.text2D(0.02, 0.92, "feature 4 (petal width) = ?",
                transform=_ax1.transAxes, fontsize=10, color="#b45309",
                bbox=dict(boxstyle="round,pad=0.35", fc="#fef3c7", ec="#f59e0b"))

    # --- right: PCA puts all four features on one 2-D map ---
    _Xs = _StandardScaler().fit_transform(_X)
    _pca = _PCA(n_components=2)
    _Xp = _pca.fit_transform(_Xs)
    _ev = _pca.explained_variance_ratio_ * 100
    _ax2 = _fig.add_subplot(1, 2, 2)
    for _i, (_c, _s) in enumerate(zip(_cols, _species)):
        _m = _y == _i
        _ax2.scatter(_Xp[_m, 0], _Xp[_m, 1], s=22, color=_c, alpha=0.85, label=_s)
    _ax2.set_xlabel("PC1  (%.0f%% of variance)" % _ev[0], fontsize=9)
    _ax2.set_ylabel("PC2  (%.0f%%)" % _ev[1], fontsize=9)
    _ax2.set_title("PCA — all 4 features in one 2-D map", fontsize=11, color=_ink)
    _ax2.tick_params(labelsize=7)

    _handles, _labels = _ax2.get_legend_handles_labels()
    _fig.legend(_handles, _labels, loc="outside lower center", ncol=3,
                frameon=False, fontsize=8)

    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.md(
                r"""## Idea 2 — PCA in 60 seconds

**iris** has 4 features, but we can only *draw* **three** axes — what about **feature 4?** ❓
**PCA** keeps the two directions of largest spread (**PC1**, **PC2**) and projects **all** the features onto them."""
            ),
            mo.image(_buf, width="700px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">9 / 13</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    import io as _io
    import numpy as _np
    import matplotlib.pyplot as _plt

    _ink = "#0f172a"
    _rng = _np.random.default_rng(1)
    _cols = ["#ef4444", "#3b82f6", "#22c55e"]
    _centers = [[0.6, 0.4], [2.2, 1.0], [1.3, 2.0]]

    _fig, _axes = _plt.subplots(1, 2, figsize=(10.5, 4.3), layout="constrained")
    for _c, _cen in zip(_cols, _centers):
        _P = _rng.normal(_cen, 0.55, size=(40, 2))
        _axes[0].scatter(_P[:, 0], _P[:, 1], s=20, color=_c, alpha=0.8)
    _axes[0].set_title("feature space (512-d): classes overlap",
                       fontsize=11, color=_ink)
    for _c, _cen in zip(_cols, _centers):
        _P = _rng.normal(_cen, 0.18, size=(40, 2))
        _axes[1].scatter(_P[:, 0], _P[:, 1], s=20, color=_c, alpha=0.8)
    _axes[1].set_title("t-SNE map (2-D): clusters separated",
                       fontsize=11, color=_ink)
    for _ax in _axes:
        _ax.set_xticks([])
        _ax.set_yticks([])
        for _s in _ax.spines.values():
            _s.set_visible(False)
        _ax.margins(0.12)
    _fig.suptitle("t-SNE keeps neighbours close", fontsize=13, color=_ink)

    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.md(
                r"""## Idea 2 (cont.) — t-SNE in 60 seconds

t-SNE ignores variance. It keeps **neighbours together**: points that are close
in the 512-d space are placed close in 2-D.

- **Perplexity** = how many neighbours each point cares about.
- Great for **seeing clusters**.
- **Careful:** the distances *between* clusters and the cluster *sizes* are
  **not** meaningful, and it is **stochastic** — trust what is stable across
  re-runs.
- **Do not predict** from the 2-D map just because it looks tidy — it is for
  looking.

That last warning is exactly what makes the *Advanced* level interesting: what
happens if you do it anyway?"""
            ),
            mo.image(_buf, width="740px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">10 / 13</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The five levels

        | level | what you do | you practise |
        |:--|:--|:--|
        | **1 · Beginner** 🟢 | recreate the t-SNE figure (Fig 5) | loading data, plotting, visualising |
        | **2 · Intermediate** 🟡 | extract features + PCA yourself, then plot | pretrained models, PCA |
        | **3 · Advanced** 🔴 | train a linear classifier on the 2-D map, draw its boundary | classification, evaluation |
        | **4 · Expert** 🔴 | compare trees / forests / MLP on the **full** features | model comparison, boundaries |
        | **5 · God-like** ⚫ | classify **live** images from the Kiruna sky cam | the whole pipeline, end to end |

        You may **start at any level**. All levels use the same data and the same
        idea; each one adds a step.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">11 / 13</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## What you get, and how to start

        For your level you receive a **folder** with:

        - a **`getting_started_*.ipynb`** notebook with the tasks and empty cells,
        - the **data** the level needs (a CSV, an image archive, or precomputed
          **features**),
        - for some levels, **models** we have already trained for you,
        - a short **README** explaining the folder.

        **To start (local):** put the folder somewhere, then

        ```bash
        uv run jupyter lab        # then open the getting_started notebook
        ```

        **In the cloud:** upload the folder to Colab / Jupyter and open the same
        notebook. Everything runs on a laptop CPU.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">12 / 13</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Over to you

        Pick a level, open the folder, and work through the getting-started
        notebook. There is no hidden test and no single "right" score — we care
        that you **look at the data**, **explain what you see**, and **evaluate
        honestly**.

        > Questions at any point — that is what these two days are for.
        > **Let's get started.**
        >
        > 📄 [Aurora project slides](https://wickstrom.github.io/eugloh-machine-learning-handbook/notebooks/03/)

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">13 / 13</div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
