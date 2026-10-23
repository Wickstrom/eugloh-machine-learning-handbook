# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
# ]
# ///
#
# On-site project — Level 4 (Expert) task slides.
# Compare logistic regression, k-NN and a random forest on the 2-D map; plot each
# model's errors (green = correct, red = wrong) and its decision regions by
# scattering a grid of points.
#
# Run locally with `marimo edit notebooks/03/06_project_expert.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).

import marimo

__generated_with = "0.17.6"
app = marimo.App(
    width="medium",
    layout_file="layouts/06_project_expert.slides.json",
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
        # Project · Level 4 — Expert

        ## Compare classifiers on the 2-D map

        🔴 **model comparison · error analysis · decision regions**

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

        Level 3 trained **one** model on the 2-D map. Now compare **three** — and
        find out what each one gets wrong.

        1. Fit **three classifiers** on the same 2-D data: **logistic regression**,
           **k-NN** and a **random forest**. The code is **your job**; the
           notebook links to the sklearn docs.
        2. **Compare** their accuracy on the test set.
        3. **Plot every model's mistakes**: one panel per model, **green** =
           correct, **red** = wrong.
        4. **Draw the decision regions** of one model by predicting a **grid** of
           points and **scattering** them — no `contourf`.

        The point: **different models fail in different places.** We stay in 2-D,
        because that is the only space we can actually draw.

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
        expert/
          getting_started_expert.ipynb   ← start here
          data/aurora_2d_train.csv       ← training images in the 2-D map
          data/aurora_2d_test.csv        ← test images in the same map
          data/dataset_info.json
          README.md
        ```

        Two CSVs, one row per image: `tsne_1`, `tsne_2` (the coordinates),
        `label` (0–6), `class_name`, and a few extra columns.

        Everything runs with **`scikit-learn`** — no PyTorch, no GPU, no 512-d
        features.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 8</div>
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

        train = pd.read_csv("data/aurora_2d_train.csv")
        test  = pd.read_csv("data/aurora_2d_test.csv")

        X_train = train[["tsne_1", "tsne_2"]].to_numpy()
        y_train = train["label"].to_numpy()
        X_test  = test[["tsne_1", "tsne_2"]].to_numpy()
        y_test  = test["label"].to_numpy()
        ```

        - one row per image: `tsne_1`, `tsne_2` are the coordinates, `label` is the class.
        - all three models get the **same** `X_train` / `y_train`.
        - **your turn:** fit logistic regression, k-NN and a forest; keep the predictions in `preds`.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 8</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The tasks (in the notebook)

        1. **Load** the 2-D coordinates.
        2. **Your turn:** fit **logistic regression, k-NN and a random forest** → `preds`.
        3. **Compare** the three accuracies.
        4. **Your turn:** one green/red error panel per model.
        5. **Your turn:** predict a **grid** of points and scatter it → decision regions.
        6. **Your turn (more):** which model has the simplest regions? The best accuracy?

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 8</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.md(r"""## The target — three models, three error patterns"""),
            mo.image(
                str(mo.notebook_location() / "public" / "aurora_errors_models.png"),
                width="860px",
            ),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">6 / 8</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Watch out

        - **Accuracy lies on imbalanced data.** *Patchy* dominates, so a lazy model
          looks good. Also report **macro F1**.
        - **More complex is not always better.** k-NN and forests can carve finer
          regions but still lose on the rare classes.
        - **Green/red hides *which* class is wrong.** Follow it with a confusion
          matrix.
        - **The t-SNE coordinates leak.** They were fitted on train and test
          together, so these scores look better than for genuinely new images.
        - **Do not tune on the test set.** Use cross-validation on the training
          data, then touch the test set **once**.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">7 / 8</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Nice work — a model comparison

        - fitted **three classifiers** on the same 2-D data,
        - compared their accuracy,
        - plotted every model's mistakes, **green/red**,
        - drew the decision regions with a **grid + scatter** (no `contourf`).

        **Done? Level 5 (God-like)** takes a trained model out of the notebook and
        points it at the **live sky**.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">8 / 8</div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
