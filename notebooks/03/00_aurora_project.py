# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
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
                str(mo.notebook_location() / "public" / "aurora_tsne_classes.png"),
                width="860px",
            ),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">6 / 13</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt
    from matplotlib.patches import Rectangle as _Rect

    _fig, _ax = _plt.subplots(figsize=(9.6, 3.1), layout="constrained")
    _ax.axis("off")
    _ax.set_xlim(0, 10)
    _ax.set_ylim(0, 3)

    _ax.add_patch(_Rect((0.2, 1.0), 1.2, 1.0, color="#e2e8f0", ec="#94a3b8"))
    _ax.text(0.8, 1.5, "image", ha="center", va="center", fontsize=9)

    _ax.add_patch(_Rect((2.0, 0.8), 2.2, 1.4, color="#dbeafe", ec="#60a5fa"))
    _ax.text(3.1, 1.75, "pretrained CNN", ha="center", fontsize=9)
    _ax.text(3.1, 1.25, "(frozen)", ha="center", fontsize=8, color="#6b7280")

    _ax.add_patch(_Rect((4.9, 0.9), 1.5, 1.2, color="#dcfce7", ec="#16a34a"))
    _ax.text(5.65, 1.5, "features", ha="center", va="center", fontsize=9)

    _ax.add_patch(_Rect((7.0, 1.85), 2.7, 0.8, color="#ede9fe", ec="#8b5cf6"))
    _ax.text(8.35, 2.25, "PCA → t-SNE  (look)", ha="center", va="center", fontsize=8.5)

    _ax.add_patch(_Rect((7.0, 0.35), 2.7, 0.8, color="#fef3c7", ec="#f59e0b"))
    _ax.text(8.35, 0.75, "classifier  (predict)", ha="center", va="center", fontsize=8.5)

    for _xy, _xytext in [((2.0, 1.5), (1.4, 1.5)), ((4.9, 1.5), (4.2, 1.5))]:
        _ax.annotate("", xy=_xy, xytext=_xytext,
                     arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#6b7280"))
    _ax.annotate("", xy=(7.0, 2.25), xytext=(6.4, 1.6),
                 arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#6b7280"))
    _ax.annotate("", xy=(7.0, 0.75), xytext=(6.4, 1.4),
                 arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#6b7280"))

    _ax.set_title("The whole project in one picture")
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.image(_buf, width="880px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">7 / 13</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Idea 1 — pretrained features

        Training an image model from scratch needs millions of labelled images
        and many GPUs. Instead we take a network already trained on a huge
        dataset (ImageNet) and **keep the layer just before its classifier**.

        - For any image that layer produces a **feature vector** — an
          **embedding** (512 or 2048 numbers).
        - **Similar images land near each other** in this space.
        - We can then feed the embedding to **any** classifier we know —
          logistic regression, a tree, a forest, a small MLP.

        In plain words: **turn each image into a compact, meaningful vector, then
        use the models from earlier in the course.**

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">8 / 13</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Idea 2 — PCA in 60 seconds

        PCA finds the directions in which the data **varies the most**.

        1. **Centre** the features (subtract the mean).
        2. Find the **directions of maximum variance** (eigenvectors of the
           covariance, equivalently the **SVD**).
        3. **Project**: keep the top directions — for us, **two**, so we can plot.

        - It is **linear, fast and deterministic** — the right first thing to try.
        - Always **standardise** first; PCA is scale-sensitive.
        - **Maximum variance is not the same as best separation** — it ignores
          the labels.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">9 / 13</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Idea 2 (cont.) — t-SNE in 60 seconds

        t-SNE ignores variance. It asks: **which points are neighbours?** — then
        lays them out in 2-D so neighbours stay close.

        - **Perplexity** sets how many neighbours each point cares about.
        - Great for **seeing** clusters.
        - **But be careful:** cluster *distances* and *sizes* are **not**
          meaningful, and it is **stochastic** — trust what is stable across
          re-runs.
        - **Do not** feed the 2-D coordinates into a classifier because they
          look pretty — this is for looking, not predicting.

        That last warning is exactly what makes the *Advanced* level interesting:
        what happens if you do it anyway?

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">10 / 13</div>
        """
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
        ## What to hand in

        A notebook (yours, based on the getting-started one) that shows:

        1. the **figures** you produced,
        2. a **short written answer** to each task — a sentence or two is enough,
        3. the **numbers** you measured (accuracy, F1, …), on the **test set**.

        There is no hidden test and no single "right" score — we care that you
        **look at the data**, **explain what you see**, and **evaluate honestly**.

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
