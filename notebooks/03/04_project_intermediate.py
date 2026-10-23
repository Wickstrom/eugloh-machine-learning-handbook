# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
# ]
# ///
#
# On-site project — Level 2 (Intermediate) task slides.
# Load the precomputed 512-d features, reduce them with PCA, and plot the 2-D map.
#
# Run locally with `marimo edit notebooks/03/04_project_intermediate.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).

import marimo

__generated_with = "0.17.6"
app = marimo.App(
    width="medium",
    layout_file="layouts/04_project_intermediate.slides.json",
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
        # Project · Level 2 — Intermediate

        ## Reduce the features with PCA

        🟡 **features · PCA**

        **Machine Learning with Python** — On-site project (Oct 23–24)

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">1 / 9</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Your mission

        In Level 1 someone else computed the 2-D map for you. Now **you** make one
        yourself.

        1. Take the **512-d feature vectors** — already computed for you (feature
           extraction is slow, so we did it once).
        2. **Standardise** them.
        3. Reduce them to **2-D with PCA** — the code is **your job**; the notebook
           links to the sklearn docs.
        4. **Plot** the result with your **Level 1 plotting code** and compare it
           with the t-SNE map.

        The point: **PCA is another way to get to 2-D.** t-SNE is for looking; PCA
        is something you can fit and reuse.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">2 / 9</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## What is in your folder

        ```
        intermediate/
          getting_started_intermediate.ipynb   ← start here
          data/features_resnet18.npz           ← 512-d features + labels (precomputed)
          data/dataset_info.json
          data/00_ReadMe.txt, data/kagolicens.pdf
          README.md
        ```

        `features_resnet18.npz` holds four arrays:

        | key | shape | meaning |
        |:----|:------|:--------|
        | `z_train` | `(3000, 512)` | training features |
        | `y_train` | `(3000,)` | training labels (0–6) |
        | `z_test` | `(846, 512)` | test features |
        | `y_test` | `(846,)` | test labels |

        You **load** these and do the **PCA yourself** — everything else is the
        plotting you already know.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 9</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## How to start

        **No PyTorch, no downloads** — just `numpy`, `matplotlib` and
        `scikit-learn`, and the features are already in the folder.

        ```bash
        uv run jupyter lab
        ```

        or upload the folder to **Google Colab**. The notebook runs in **seconds**.

        You will **write the PCA code yourself**. The notebook links to the
        sklearn docs:
        <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 9</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The code — loading the features

        ```python
        import numpy as np

        data = np.load("data/features_resnet18.npz")

        z_train = data["z_train"]      # (3000, 512)
        y_train = data["y_train"]
        z_test  = data["z_test"]       # (846, 512)
        y_test  = data["y_test"]
        ```

        - each image is already a **512-number vector** — no extraction needed.
        - `z_train`/`y_train` are the training set, `z_test`/`y_test` the test set.
        - from here it is **your turn**: `PCA(n_components=2)`, then plot.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 9</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The tasks (in the notebook)

        1. **Load** the 512-d features with `np.load(...)`.
        2. **Your turn:** make a `PCA(n_components=2)` and `fit_transform` the
           training features. The notebook links to the sklearn docs.
        3. **Plot** the 2-D result with your **Level 1 plotting code**.
        4. **Check** `pca.explained_variance_ratio_` — how much did the two PCs keep?
        5. **Your turn (more):** plot the test points too.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">6 / 9</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.md(r"""## What it looks like — PCA of the features"""),
            mo.image(
                str(mo.notebook_location() / "public" / "aurora_pca_2d.png"),
                width="780px",
            ),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">7 / 9</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Watch out

        - **Standardise before PCA.** A feature with a bigger range would
          otherwise dominate. `sklearn`'s `StandardScaler` is your friend.
        - **PCA is linear.** It finds the directions of most *variance*, not the
          directions that best *separate the classes*. Two classes can sit on top
          of each other in the PCA plot and still be separable.
        - **The two largest directions are not "the best" two** — they are just
          the two with most spread.
        - Two PCs rarely explain everything: look at how much variance you kept.

        That last point is a **cliffhanger** for the next level: if the 2-D map
        throws information away, how well can a classifier do on it?

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">8 / 9</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Nice work — you did PCA yourself

        - loaded the precomputed **512-d features**,
        - standardised them and reduced them to **2-D with PCA**,
        - plotted the map with your **Level 1 plotting code**.

        **Done? Try Level 3 (Advanced)** — train a classifier on the 2-D map and
        find the images it gets wrong.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">9 / 9</div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
