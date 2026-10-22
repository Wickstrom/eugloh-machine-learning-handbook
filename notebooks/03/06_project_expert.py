# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
# ]
# ///
#
# On-site project — Level 4 (Expert) task slides.
# Compare several models (tree, forest, MLP, ...) on the full 512-d features and
# on the 2-D map, and draw decision boundaries in the 2-D case.
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

        ## Compare models on the full features and on the 2-D map

        🔴 **model comparison · decision boundaries**

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

        Level 3 used a **single linear** model on the **2-D** map. Now you run a
        proper comparison.

        1. Take **two representations** of every image:
           the **full 512-d ResNet features**, and the **2-D map**.
        2. Train a **range of models** on each: logistic regression, **decision
           tree**, **random forest**, **k-NN**, and a small **MLP**.
        3. **Compare** them with **accuracy** and **macro F1**.
        4. **Draw the decision boundaries** for the tree, forest and MLP on the
           2-D map — and see how they differ from a straight line.

        This is the whole supervised-learning toolkit from the course, applied to
        new features.

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
        ## The code — one model, two representations

        ```python
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score

        model = LogisticRegression(max_iter=5000, class_weight="balanced")

        model.fit(z_train, y_train)                    # 512-d features
        print("512-d:", accuracy_score(y_test, model.predict(z_test)))

        model.fit(X_train2, y_train)                   # 2-D map
        print("2-D  :", accuracy_score(y_test, model.predict(X_test2)))
        ```

        - the **same model** is fitted twice — only the input changes.
        - `z_*` are the **512-d features**; `X_*2` are the two t-SNE columns.
        - the gap between the two numbers is what the **2-D map costs us**.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 8</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The tasks (in the notebook)

        1. **Load** the 512-d features (`features_resnet18.npz`) and the 2-D coordinates.
        2. **Train** the **same model twice**: once on the 512-d features, once on the 2-D map.
        3. **Compare** the two accuracies — how much does 2-D cost?
        4. **Your turn:** swap in a random forest or k-NN. Does it beat logistic regression?
        5. **Your turn:** which representation wins, and by how much?

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 8</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.md(r"""## What it looks like — one scoreboard, two representations"""),
            mo.image(
                str(mo.notebook_location() / "public" / "aurora_model_comparison.png"),
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

        - **Macro F1, not accuracy.** With *Patchy* dominating, accuracy flatters
          every model. Macro F1 exposes the models that ignore the rare classes.
        - **Trees overfit the 2-D map** — they can carve the plane into thousands
          of tiny boxes. Limit `max_depth` (or `min_samples_leaf`) and watch the
          boundary get simpler.
        - **MLPs need scaling.** Standardise the 512-d features first, or the
          network trains badly. Also set a seed (`random_state=0`).
        - **k-NN is lazy.** No training, but prediction is slower and it needs the
          whole training set.
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

        - trained the **same model** on the **512-d features** and on the **2-D map**,
        - measured how much accuracy the 2-D map costs,
        - tried a second model type. Which won?

        **Done? Level 5 (God-like)** takes the trained model out of the notebook
        and points it at the **live sky**.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">8 / 8</div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
