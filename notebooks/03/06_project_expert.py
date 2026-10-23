# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
# ]
# ///
#
# On-site project — Level 4 (Expert) task slides.
# Fit the same model on the 512-d features and on the 2-D map, then plot both
# models' errors on the map (green = correct, red = wrong).
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

        ## One model, two inputs — and where each fails

        🔴 **model comparison · error analysis**

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

        Level 3 used only the **2-D map**. Now compare it with the **full 512-d
        features**.

        1. Take **two representations** of every image: the **512-d ResNet
           features** and the **2-D t-SNE map**.
        2. Fit the **same model** on each — a logistic regression. The code is
           **your job**; the notebook links to the sklearn docs.
        3. **Compare** the two accuracies: how much does the 2-D map cost us?
        4. **Plot both models' mistakes on the same map**: **green** = correct,
           **red** = wrong. Where does each representation fail?

        The point: **the representation matters as much as the model.**

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
          data/features_resnet18.npz     ← 512-d features + labels (precomputed)
          data/aurora_2d_train.csv       ← the same images in the 2-D map
          data/aurora_2d_test.csv
          data/dataset_info.json
          README.md
        ```

        `features_resnet18.npz` holds `z_train`, `y_train`, `z_test`, `y_test`
        (features are `(3000, 512)` and `(846, 512)`).

        Everything runs with **`scikit-learn`** — no PyTorch, no GPU.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 8</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The code — loading both representations

        ```python
        import numpy as np
        import pandas as pd

        f = np.load("data/features_resnet18.npz")
        z_train, y_train = f["z_train"], f["y_train"]
        z_test,  y_test  = f["z_test"],  f["y_test"]

        d2_train = pd.read_csv("data/aurora_2d_train.csv")
        d2_test  = pd.read_csv("data/aurora_2d_test.csv")
        X_train2 = d2_train[["tsne_1", "tsne_2"]].to_numpy()
        X_test2  = d2_test[["tsne_1", "tsne_2"]].to_numpy()
        ```

        - `z_*` are the **512-d features**; `X_*2` are the **2-D coordinates** of the same images.
        - the labels are shared, so the two representations are directly comparable.
        - **your turn:** fit a logistic regression on each and keep `pred_512` / `pred_2d`.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 8</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The tasks (in the notebook)

        1. **Load** the 512-d features and the 2-D coordinates.
        2. **Your turn:** fit the **same model twice** (512-d and 2-D) → `pred_512`, `pred_2d`.
        3. **Compare** the two accuracies — how much does 2-D cost?
        4. **Your turn:** plot both models' errors on the map, **green** = correct, **red** = wrong.
        5. **Your turn (more):** swap in a random forest or k-NN.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 8</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.md(r"""## The target — the same map, two error patterns"""),
            mo.image(
                str(mo.notebook_location() / "public" / "aurora_errors_2d_vs_512.png"),
                width="820px",
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
        - **Same model, different representation.** If 512-d wins, that is the
          **representation**, not a better model.
        - **Green/red hides *which* class is wrong.** Follow it with a confusion
          matrix.
        - **The t-SNE coordinates leak.** They were fitted on train and test
          together, so the 2-D scores look better than for genuinely new images.
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
        ## Nice work — comparing two representations

        - fitted the **same model** on the **512-d features** and on the **2-D map**,
        - measured how much accuracy the 2-D map costs,
        - plotted both models' errors on the same map, **green/red**,
        - saw **where** each representation fails.

        **Done? Level 5 (God-like)** takes a trained model out of the notebook and
        points it at the **live sky**.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">8 / 8</div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
