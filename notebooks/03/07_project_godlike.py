# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
# ]
# ///
#
# On-site project — Level 5 (God-like) task slides.
# Fetch live images from the IRF Kiruna all-sky camera, extract features, and
# classify them with a trained model — a small "is there aurora right now?" tool.
#
# Run locally with `marimo edit notebooks/03/07_project_godlike.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).

import marimo

__generated_with = "0.17.6"
app = marimo.App(
    width="medium",
    layout_file="layouts/07_project_godlike.slides.json",
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
        # Project · Level 5 — God-like

        ## Classify the live sky over Kiruna

        ⚫ **the whole pipeline, end to end, on real data you fetch yourself**

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

        Everything so far ran on the **2010–2019 dataset**. Now use what you built
        on images that were taken **minutes ago**.

        1. **Fetch** recent images from the **IRF Kiruna all-sky camera**.
        2. **Preprocess** them the way the training images were prepared
           ($128\times128$).
        3. **Extract features** with the same pretrained ResNet-18 you used before.
        4. **Classify** each image with a model we trained for you.
        5. Turn it into a tiny **nowcast**: *is there aurora over Kiruna right
           now — and what kind?*

        This is exactly how research pipelines get built: **train once, then run it
        on new data.**

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
        godlike/
          getting_started_godlike.ipynb    ← start here
          aurora_features.py               ← the ResNet-18 feature extractor
          fetch_skycam.py                  ← downloads recent Kiruna all-sky images
          models/aurora_classifier.joblib  ← a classifier trained on the 512-d features
          README.md
        ```

        - `aurora_classifier.joblib` predicts both the **7 classes** and, through
          it, **aurora vs not aurora**.
        - It was trained on the **same** pretrained features you extract, so the
          two must match — use `aurora_features.py` as-is.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 8</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## How to start

        This level needs **PyTorch** (for the feature extractor) and **internet**
        (to fetch the images):

        ```bash
        uv pip install torch torchvision pillow requests
        uv run jupyter lab
        ```

        **Easiest — Google Colab:** `torch` and `requests` are already there; just
        upload the folder.

        The camera you are reading from — browse it here:
        `https://firmament.irf.se/`. It publishes one image per minute, and
        `fetch_skycam.py` talks to the same public archive the viewer uses.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 8</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The tasks (in the notebook)

        1. **Fetch** the latest image (and a few recent ones) with `fetch_skycam.py`.
           Display them.
        2. **Preprocess** — this is the point. The model wants a $128\times128$
           **crop**, not the whole fisheye. Cut a window that matches the training
           crops (central 50% by default) and try a few zooms.
        3. **Extract features** with `aurora_features.extract_embeddings(...)`.
        4. **Classify** with the provided model. Show the predicted class, the
           probability of each class, and a simple **"aurora / not aurora"** flag.
        5. **Build a nowcast:** loop over the last few images and print a one-line
           verdict per image, with the time.
        6. **Think like a researcher:** how would you *trust* this? What would you
           need to check before believing it?

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 8</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Watch out — this is the hard level, and that is the point

        - **Crop, don't just resize.** The model learned from **labelled
          $128\times128$ zoomed-in night crops**. A live frame is a whole-sky
          **fisheye** (480x480, black disk edge plus a timestamp/compass overlay).
          Feeding it the full disk is out of distribution — so **choosing the crop
          is the whole task.** Start with the central 50% and experiment.
        - **URLs change.** The "latest" image is overwritten every minute; grab it,
          don't cache it, and handle a failed download gracefully.
        - **No labels for live images.** You cannot compute accuracy here — so you
          must reason about whether the predictions *look* right.
        - **Be honest about uncertainty.** A confident wrong answer is worse than
          flagging an image the model clearly cannot handle.

        That is why this level is worth a **whole section of your write-up**: the
        result matters less than the reasoning.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">6 / 8</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Stretch ideas

        - **Time series.** Fetch a stretch of the night (many frames from the day's
          manifest) and watch auroral activity rise and fall.
        - **Compare nights.** Pick a quiet night and a stormy night from the
          archive and classify both.
        - **Train your own head.** Re-extract features for the training set and
          train your **own** classifier on top (you have `data.npz` in Level 2),
          then compare it with ours.
        - **Calibration.** Are the model's probabilities trustworthy? Plot a
          reliability curve on the test set.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">7 / 8</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Nice work — the whole arc, end to end

        You took the project all the way:

        - fetched **live images** and preprocessed them,
        - produced a **predicted class** and an **aurora / not-aurora** verdict
          per image,
        - built a **nowcast**,
        - judged **how much to trust it** — the most important skill of all.

        You have now done the whole arc: **raw image → features → model →
        prediction on the live sky.** Well done. 🌌

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">8 / 8</div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
