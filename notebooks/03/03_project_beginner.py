# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
# ]
# ///
#
# On-site project — Level 1 (Beginner) task slides.
# A short, self-contained "how to get started" deck for the entry-level project:
# load the precomputed 2-D coordinates, plot the t-SNE map, and rebuild a
# Figure-5-style picture with example images drawn on the map.
#
# Run locally with `marimo edit notebooks/03/03_project_beginner.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).

import marimo

__generated_with = "0.17.6"
app = marimo.App(
    width="medium",
    layout_file="layouts/03_project_beginner.slides.json",
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
        # Project · Level 1 — Beginner

        ## Recreate the t-SNE map of the aurora images

        🟢 **loading · plotting · visualising**

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

        We have already done the hard part for you. Every image was pushed through
        a **pretrained network** and its **features were projected to two
        dimensions with t-SNE**. All you have to do is **look at the result**.

        You will make **one figure** — a map where

        - every point is **one test image**,
        - the **colour** is its **aurora class**, and
        - a few **example images** are drawn on top of the map.

        That is exactly the picture from **Figure 5** of the paper.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">2 / 9</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.md(r"""## The target — something like this"""),
            mo.image(
                str(mo.notebook_location() / "public" / "aurora_fig5_reference.png"),
                width="720px",
            ),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 9</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## What is in your folder

        ```
        beginner/
          getting_started_beginner.ipynb   ← start here
          data/
            aurora_2d_test.csv             ← the 2-D coordinates (846 test images)
            aurora_2d_train.csv            ← optional: the 3000 training images
            dataset_info.json              ← class names, counts, …
          images/test/                     ← the 846 test images (PNG)
          README.md
        ```

        Each **row** of the CSV is one image:

        `filename`, `tsne_1`, `tsne_2`, `label`, `class_name`, `aurora`, `split`

        Open `getting_started_beginner.ipynb` and work through it in order.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 9</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## How to start

        1. Put the folder somewhere you can find it.
        2. Open a terminal **in that folder** and run:

           ```bash
           uv run jupyter lab
           ```

           (no `uv`? any Jupyter works — `pandas` + `matplotlib` is all you need)
        3. In Jupyter, open **`getting_started_beginner.ipynb`**.
        4. Run the first cells — they load the data and draw a first scatter plot.
        5. Then work through the **Tasks**. Fill in the empty cells.

        **Nothing to install, nothing to train** — this level runs in seconds.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 9</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The code — loading the map

        ```python
        import pandas as pd
        import matplotlib.pyplot as plt

        test = pd.read_csv("data/aurora_2d_test.csv")
        test.head()
        ```

        - `pd.read_csv` reads the file into a table (a **DataFrame**).
        - every **row** is one test image: its `tsne_1`, `tsne_2` and its class.
        - `test.head()` shows the first rows so you can check the columns.

        After that you only **scatter** `tsne_1` vs `tsne_2`. The notebook leaves
        the colouring for **you** to fill in.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">6 / 9</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The tasks (in the notebook)

        1. **Load** `data/aurora_2d_test.csv` and look at `head()`.
        2. **Count** the images per class with `value_counts()`.
        3. **Plot** the map: `tsne_1` vs `tsne_2` (a starting cell is given).
        4. **Colour** the points by class and add a legend — **your turn**: fill in the blank.
        5. **Overlay** an example image on the map — one is worked out for you, add more.
        6. Answer the short **Your turn** questions.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">7 / 9</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Watch out — how to read a t-SNE plot

        - **Only the grouping is meaningful.** The distance between two clusters
          does **not** mean "how different" they are.
        - **Cluster sizes are not meaningful** either.
        - **It is random** — you may see tiny differences each run. Trust what is
          stable.
        - Some classes **overlap** (look at *Discrete* / *Colored* / *Edge*).
          **That overlap is real** — it is exactly why classification is hard.

        Read that last point again: the map is not just pretty, it **explains the
        mistakes** the models will make later.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">8 / 9</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Nice work — you rebuilt the map

        - loaded the precomputed **2-D coordinates** and the **labels**,
        - coloured the scatter by **aurora class**,
        - placed an **example image** onto the map.

        **Done? Try Level 2 (Intermediate)** — there you extract the features
        yourself.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">9 / 9</div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
