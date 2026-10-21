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

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">1 / 8</div>
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

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">2 / 8</div>
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
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 8</div>"""),
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

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 8</div>
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

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 8</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The tasks (in the notebook)

        1. **Load** the data and print its shape, its columns, and the class counts.
        2. **Plot** the t-SNE map — `tsne_1` vs `tsne_2` — one colour per class, with a legend.
        3. **Show** a grid of example images, a few per class.
        4. **Overlay** a few images on the map: pick a point, load its PNG, draw it as a small inset near the point. *(This rebuilds Figure 5.)*
        5. **Label** the clusters — add text near each group naming the class.
        6. **Answer** the short wrap-up questions about what you notice.

        Take your time on 2 and 4 — those are the heart of the level.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">6 / 8</div>
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

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">7 / 8</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## What to hand in

        Your notebook, containing:

        - the **t-SNE map** coloured by class,
        - the map with **example images overlaid** (your Figure 5),
        - the **grid of example images**,
        - your answers to the wrap-up questions.

        Keep it simple and readable. **Done — then try Level 2 (Intermediate)**,
        where you extract the features yourself.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">8 / 8</div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
