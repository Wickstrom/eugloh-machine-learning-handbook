# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
# ]
# ///
#
# On-site project — Level 3 (Advanced) task slides.
# Train a linear classifier on the 2-D t-SNE coordinates, draw its decision
# boundary, and evaluate it.
#
# Run locally with `marimo edit notebooks/03/05_project_advanced.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).

import marimo

__generated_with = "0.17.6"
app = marimo.App(
    width="medium",
    layout_file="layouts/05_project_advanced.slides.json",
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
        # Project · Level 3 — Advanced

        ## A linear classifier on the 2-D map

        🔴 **classification · decision boundary · evaluation**

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

        You have the **2-D t-SNE map** (Level 1) and you know that **t-SNE is for
        looking, not for predicting**. In this level you ignore that warning on
        purpose — and measure the cost.

        1. Train a **logistic regression** on the 2-D coordinates: predict the
           aurora class from `(tsne_1, tsne_2)`.
        2. **Draw the decision boundary** on the map.
        3. **Evaluate**: accuracy, per-class F1, and a **confusion matrix**.
        4. Ask: how much does the 2-D map *lose* compared with the full features?

        This connects the **supervised models from Day 1** (logistic regression,
        trees) to the new features from Day 3.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">2 / 8</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## What is in your folder

        ```
        advanced/
          getting_started_advanced.ipynb   ← start here
          data/aurora_2d_train.csv         ← 3000 training points
          data/aurora_2d_test.csv          ← 846 test points
          data/dataset_info.json
          README.md
        ```

        Columns: `filename`, `tsne_1`, `tsne_2`, `label`, `class_name`, `aurora`,
        `aurora_name`, `split`.

        Just `pandas`, `matplotlib`, `scikit-learn` — **no GPU, no downloads**.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 8</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The tasks (in the notebook)

        1. **Load** the train/test points and plot them coloured by class.
        2. **Baseline:** always predict the majority class — what accuracy (and
           why is accuracy misleading here)?
        3. **Logistic regression** on `(tsne_1, tsne_2)`. Report **accuracy** and
           **macro F1** and print the **classification report**.
        4. **Draw the decision boundary**: predict on a fine grid and use
           `contourf`, then scatter the true points on top.
        5. **Confusion matrix** — which two classes are confused most? Do they
           overlap in the map? *(They will.)*
        6. **Reflect:** the boundary is made of **straight lines**. What shape of
           region would a decision **tree** make instead?

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 8</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.md(r"""## The target — a linear boundary on the 2-D map"""),
            mo.image(
                str(mo.notebook_location() / "public" / "aurora_logreg_boundary.png"),
                width="680px",
            ),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 8</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Watch out

        - **Accuracy lies on imbalanced data.** *Patchy* is ~46% of the test set,
          so a lazy model looks good. Use **macro F1** (every class counts
          equally) and read the **per-class** F1.
        - **The t-SNE coordinates are not real features.** They were fitted on
          *all* the data (train and test together), which leaks information and
          makes the score look better than it would be for genuinely new images.
          Worth a sentence in your write-up.
        - **A linear boundary cannot carve out islands.** Some classes live in
          disconnected blobs — a straight-line model must fail on them.
        - Set a **random seed** (`random_state=0`) so your numbers are repeatable.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">6 / 8</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Stretch (optional)

        - Compare **`class_weight="balanced"`** on and off. Which do you prefer,
          and why?
        - Try a **k-nearest-neighbour** classifier on the same 2-D points.
          Non-linear! Does it beat logistic regression?
        - Do the whole thing for the **binary** problem (*aurora* vs *not
          aurora*, using the `aurora` column). Is it easier than 7 classes?

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

        - the **scatter plot** and the **decision-boundary** figure,
        - the **confusion matrix**,
        - **accuracy and macro F1** for the baseline and for logistic regression,
        - one short paragraph: *how much does the 2-D map cost us?*

        Done? **Level 4 (Expert)** uses the **full 512-d features** and compares
        several models.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">8 / 8</div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
