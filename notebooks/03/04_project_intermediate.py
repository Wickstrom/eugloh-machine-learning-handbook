# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
# ]
# ///
#
# On-site project — Level 2 (Intermediate) task slides.
# Extract features from the raw images with a pretrained network, reduce them
# with PCA, and plot the 2-D map.
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

        ## Extract the features yourself

        🟡 **pretrained models · PCA**

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

        In Level 1 someone else computed the features for you. Now **you** do it.

        1. Take the **raw images** ($128\times128$, the archive you were given).
        2. Push them through a **pretrained ResNet-18** and keep the last layer
           before the classifier → a **512-number feature vector** per image.
        3. **Standardise** the features and reduce them to **2-D with PCA**.
        4. **Plot** the result and compare it with the t-SNE map from the lecture.

        The point: **see the features with your own hands** — where they come from,
        and what PCA does to them.

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
          aurora_features.py                   ← loads the ResNet and extracts features
          data/data.npz                        ← images + labels (train & test)
          data/00_ReadMe.txt, data/kagolicens.pdf
          README.md
        ```

        `data.npz` holds four arrays:

        | key | shape | meaning |
        |:----|:------|:--------|
        | `a` | `(3000, 3, 128, 128)` | training images |
        | `b` | `(3000,)` | training labels (0–6) |
        | `c` | `(846, 3, 128, 128)` | test images |
        | `d` | `(846,)` | test labels |

        You will call `aurora_features.extract_embeddings(...)` — everything else
        is the same plotting you already know.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 9</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## How to start — pick one

        **Easiest — Google Colab.** `torch` is already installed. Upload the
        folder, open the notebook, done.

        **Local.** This level needs **PyTorch** (one extra dependency):

        ```bash
        uv pip install torch torchvision
        uv run jupyter lab
        ```

        The **first** run downloads the ResNet-18 weights (about 45 MB) — after
        that it works offline. Extracting features for all 3846 images takes a
        **few minutes on a laptop CPU**.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 9</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The code — extracting features

        ```python
        from aurora_features import load_data, extract_embeddings, CLASS_NAMES

        x_train, y_train, x_test, y_test = load_data("data/data.npz")

        z_train = extract_embeddings(x_train)   # (3000, 512)
        z_test  = extract_embeddings(x_test)    # (846, 512)
        ```

        - `load_data` gives you the **images** (`x`) and **labels** (`y`), train and test.
        - `extract_embeddings` runs the pretrained **ResNet-18** and returns **512 numbers per image**.
        - after that it is just `StandardScaler` → `PCA(n_components=2)` → scatter.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 9</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The tasks (in the notebook)

        1. **Load** the images with `load_data("data/data.npz")` and show a few.
        2. **Extract** the 512-d features with the pretrained ResNet — one line:
           `extract_embeddings(x_train)`.
        3. **Standardise**, then run **PCA to 2-D**.
        4. **Scatter** the result, coloured by class.
        5. **Your turn:** how much variance do the two PCs keep? Plot the **test** points.

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
        ## Nice work — you extracted features yourself

        - loaded the **images** and labels,
        - ran the pretrained **ResNet-18** to get **512-d features**,
        - compressed them to **2-D with PCA** and plotted the map.

        **Done? Try Level 3 (Advanced)** — put a classifier on the 2-D map and
        draw its decision boundary.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">9 / 9</div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
