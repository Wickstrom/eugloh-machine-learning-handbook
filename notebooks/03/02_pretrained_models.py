# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
# ]
# ///
#
# Friday on-site session (Oct 23) — Lecture 2: Pretrained models.
# From simple ImageNet classifiers to foundation models: what "pretrained"
# means, feature extraction ("linear probing"), the difference from
# fine-tuning, zero-/few-shot with foundation models, and how pretrained
# embeddings combine with the dimensionality reduction from Lecture 1 and the
# supervised models from earlier in the course.
# Run locally with `marimo edit notebooks/03/02_pretrained_models.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).
#
# NOTE: this is a first draft. The dataset-specific section and the hands-on
# demos (which need a deep-learning stack such as torch, and therefore cannot
# run in the browser) are deliberately left as placeholders for now.

import marimo

__generated_with = "0.17.6"
app = marimo.App(
    width="medium",
    layout_file="layouts/02_pretrained_models.slides.json",
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
        # Pretrained Models

        ## From ImageNet to foundation models

        **Machine Learning with Python** — On-site session (Oct 23)

        EUGLOH — *Problem Solving Using Open-Source Languages; R and Python*

        University of Novi Sad

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">1 / 16</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Today's session

        - **Why** use someone else's trained network?
        - **Feature extraction** — a pretrained model as a powerful feature map
        - **Linear probing vs fine-tuning** — the two ways to reuse it
        - **The course dataset** — how we will process it (details to come)
        - **Foundation models** — ViT, CLIP, and zero-/few-shot learning
        - **Putting it together** with PCA/t-SNE and the models you already know

        On-site session — the second of two today.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">2 / 16</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Part 1 — Why pretrained models?

        Training a good image model **from scratch** needs a huge amount of
        data and compute: millions of labelled images, many GPUs, and days of
        training. You probably have none of that.

        But the features such a model learned are **general**: edges, textures,
        shapes, object parts. They are useful far beyond the exact task the
        model was trained on.

        - **Free, strong features.** Take a network trained on a large dataset and reuse its internal representation.
        - **Works with little data.** A simple classifier on top of good features often beats a complex model on raw pixels.
        - **The modern default.** Almost nobody trains vision models from scratch — they **transfer**.

        In plain words: **stand on the shoulders of a model that already learned to see.**

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 16</div>
        """
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt

    _milestones = [
        (1998, "LeNet"),
        (2012, "AlexNet"),
        (2014, "VGG"),
        (2015, "ResNet"),
        (2020, "ViT"),
        (2021, "CLIP"),
        (2023, "foundation\nmodels"),
    ]

    _fig, _ax = _plt.subplots(figsize=(9, 2.8), layout="constrained")
    _ax.axis("off")
    _ax.set_xlim(1996, 2026)
    _ax.set_ylim(-1, 1)
    _ax.annotate("", xy=(2026, 0), xytext=(1997, 0),
                 arrowprops=dict(arrowstyle="-|>", color="#6b7280", lw=2))
    for _i, (_yr, _name) in enumerate(_milestones):
        _up = _i % 2 == 0
        _ax.plot([_yr], [0], "o", color="#2563eb", zorder=3)
        _ax.annotate(_name, xy=(_yr, 0),
                     xytext=(_yr, 0.6 if _up else -0.6),
                     ha="center", va="center", fontsize=9, color="#111827",
                     arrowprops=dict(arrowstyle="-", color="#cbd5e1"))
    _ax.set_title("A short history of pretrained vision models", pad=12)
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.image(_buf, width="820px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 16</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## What do the layers learn?

        A deep network builds its representation **hierarchically**:

        - **Early layers** see colours and **edges** — simple, local patterns.
        - **Middle layers** combine them into **textures and parts** (corners,
          wheels, eyes).
        - **Late layers** encode **objects and concepts** — "a dog", "a car".

        Because the late layers are already about *things*, their activations are
        an excellent description of an image — one that transfers to new tasks.

        **ImageNet** (≈1.2 million labelled images, 1000 classes) is the dataset
        that made this practical: a model that is good on ImageNet has learned a
        broadly useful vocabulary of visual features.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 16</div>
        """
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt
    from matplotlib.patches import Rectangle as _Rect

    _fig, _ax = _plt.subplots(figsize=(7.6, 4.0), layout="constrained")
    _ax.axis("off")
    _ax.set_xlim(0, 1)
    _ax.set_ylim(0, 1)
    _levels = [
        ("pixels / edges", 0.12, 0.80, "#bfdbfe"),
        ("textures / parts", 0.50, 0.52, "#60a5fa"),
        ("objects / concepts", 0.88, 0.28, "#1d4ed8"),
    ]
    for _lab, _y, _w, _c in _levels:
        _ax.add_patch(_Rect((0.5 - _w / 2, _y - 0.11), _w, 0.22, color=_c, zorder=2))
        _ax.text(0.5, _y, _lab, ha="center", va="center", zorder=3, fontsize=10,
                 color="white" if _c == "#1d4ed8" else "#0b1220")
    _ax.annotate("", xy=(0.06, 0.92), xytext=(0.06, 0.06),
                 arrowprops=dict(arrowstyle="-|>", color="#6b7280", lw=2))
    _ax.text(0.10, 0.5, "deeper = more abstract", rotation=90,
             va="center", fontsize=8, color="#6b7280")
    _ax.set_title("The feature hierarchy inside a deep network")
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.image(_buf, width="680px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">6 / 16</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Feature extraction — the key idea

        Chop the **classifier head** off a pretrained network and keep the layer
        just before it. For every image, that layer produces a **feature vector**
        (an **embedding**) — perhaps 512 or 1024 numbers that summarise the image.

        - **Similar images land near each other** in embedding space.
        - The embedding was learned on a huge dataset, so it already captures
          shape, texture and semantics.
        - You can now use **any** classifier you have learned — logistic
          regression, a random forest, a small MLP — on these embeddings instead
          of on raw pixels.

        In plain words: **turn each image into a compact, meaningful vector, then
        apply what you already know.**

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">7 / 16</div>
        """
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt
    from matplotlib.patches import Rectangle as _Rect

    _fig, _ax = _plt.subplots(figsize=(9.2, 2.9), layout="constrained")
    _ax.axis("off")
    _ax.set_xlim(0, 10)
    _ax.set_ylim(0, 3)

    _ax.add_patch(_Rect((0.2, 0.8), 1.4, 1.4, color="#e2e8f0", ec="#94a3b8"))
    _ax.text(0.9, 1.5, "image", ha="center", va="center", fontsize=9)

    _ax.add_patch(_Rect((2.5, 0.7), 2.5, 1.6, color="#dbeafe", ec="#60a5fa"))
    _ax.text(3.75, 1.75, "pretrained CNN", ha="center", fontsize=9)
    _ax.text(3.75, 1.2, "(frozen)", ha="center", fontsize=8, color="#6b7280")

    for _i in range(6):
        _ax.add_patch(_Rect((5.7 + _i * 0.28, 1.0), 0.22, 0.9,
                            color="#16a34a", alpha=0.85))
    _ax.text(6.4, 2.15, "embedding vector", ha="center", fontsize=9)

    _ax.add_patch(_Rect((7.9, 1.0), 1.7, 0.9, color="#fde68a", ec="#f59e0b"))
    _ax.text(8.75, 1.45, "classifier", ha="center", fontsize=9)

    _ax.annotate("", xy=(2.5, 1.5), xytext=(1.6, 1.5),
                 arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#6b7280"))
    _ax.annotate("", xy=(5.7, 1.45), xytext=(5.0, 1.45),
                 arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#6b7280"))
    _ax.annotate("", xy=(7.9, 1.45), xytext=(7.45, 1.45),
                 arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#6b7280"))

    _ax.set_title("Feature extraction: reuse the network as a feature map")
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.image(_buf, width="840px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">8 / 16</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Two ways to reuse a pretrained model

        | | **Linear probing** (feature extraction) | **Fine-tuning** |
        |---|---|---|
        | what changes | only the new classifier | the whole network |
        | data needed | little | more |
        | compute | seconds–minutes | much more |
        | risk | underfits if features are a poor match | overfits / forgets if data is small |
        | when | your default first try | features are a poor match and you have data |

        Start with **linear probing** (just logistic regression on the
        embeddings) — it is cheap, robust, and surprisingly strong. Reach for
        **fine-tuning** only when it is clearly not enough.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">9 / 16</div>
        """
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt
    from matplotlib.patches import Rectangle as _Rect

    _fig, _ax = _plt.subplots(figsize=(9.5, 3.2), layout="constrained")
    _ax.axis("off")
    _ax.set_xlim(0, 10)
    _ax.set_ylim(0, 3)

    _ax.add_patch(_Rect((0.2, 1.1), 1.2, 0.9, color="#e2e8f0", ec="#94a3b8"))
    _ax.text(0.8, 1.55, "images", ha="center", va="center", fontsize=9)

    _ax.add_patch(_Rect((2.0, 0.9), 2.2, 1.3, color="#dbeafe", ec="#60a5fa"))
    _ax.text(3.1, 1.75, "pretrained CNN", ha="center", fontsize=9)
    _ax.text(3.1, 1.25, "(frozen)", ha="center", fontsize=8, color="#6b7280")

    _ax.add_patch(_Rect((4.9, 0.9), 1.5, 1.3, color="#dcfce7", ec="#16a34a"))
    _ax.text(5.65, 1.55, "embeddings", ha="center", va="center", fontsize=9)

    _ax.add_patch(_Rect((7.0, 1.9), 2.7, 0.75, color="#ede9fe", ec="#8b5cf6"))
    _ax.text(8.35, 2.28, "PCA → t-SNE  (visualise)", ha="center", va="center", fontsize=8.5)

    _ax.add_patch(_Rect((7.0, 0.45), 2.7, 0.75, color="#fef3c7", ec="#f59e0b"))
    _ax.text(8.35, 0.83, "classifier  (predict)", ha="center", va="center", fontsize=8.5)

    _ax.annotate("", xy=(2.0, 1.55), xytext=(1.4, 1.55),
                 arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#6b7280"))
    _ax.annotate("", xy=(4.9, 1.55), xytext=(4.2, 1.55),
                 arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#6b7280"))
    _ax.annotate("", xy=(7.0, 2.28), xytext=(6.4, 1.6),
                 arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#6b7280"))
    _ax.annotate("", xy=(7.0, 0.83), xytext=(6.4, 1.5),
                 arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#6b7280"))

    _ax.set_title("The pipeline: pretrained features, then visualise or predict")
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.image(_buf, width="860px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">10 / 16</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Part 2 — The course dataset

        > **Placeholder — details to be confirmed.** This section will introduce
        > the dataset we use in the hands-on part and walk through the pipeline
        > on it.

        The plan for the hands-on part:

        1. **Look** at the data — what are the inputs, how many classes and samples?
        2. **Prepare** it — resize, normalise, and split train / validation / test.
        3. **Extract** a pretrained embedding for every sample (frozen network).
        4. **Visualise** the embeddings with **PCA / t-SNE** (Lecture 1).
        5. **Classify** them with logistic regression / random forest and compare against a baseline.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">11 / 16</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Part 3 — Foundation models

        Pretrained models have grown from **task-specific** (an ImageNet
        classifier) to **general-purpose** — models whose representations work
        across many tasks and even across modalities.

        - **Vision Transformer (ViT)** — split the image into patches and apply
          the same attention mechanism as language models.
        - **CLIP** — trained on 400M *image–text* pairs, it places images and
          captions in **one shared space**, so you can match an image to text.
        - **Self-supervised** models (e.g. **DINO**) — learn from images *without
          labels*; the embeddings are excellent and very transferable.
        - **Segment Anything (SAM)** — promptable segmentation; a foundation
          model for masks.
        - **Large language models** — the text-side analog: pretrain once, adapt
          to almost anything.

        The pattern is the same as before — **pretrain on a lot, reuse broadly** —
        just at a much larger scale.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">12 / 16</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Zero-shot and few-shot

        **Zero-shot** — with CLIP, no training examples are needed at all. Write
        the class names as text prompts, embed them, and assign each image the
        class whose prompt is closest:

        > "a photo of a **cat**" · "a photo of a **dog**" · "a photo of a **bird**"

        Because images and text live in the same space, this often works out of
        the box.

        **Few-shot** — a handful of labelled examples per class, then a **linear
        probe** on the frozen embeddings. A few images often go a long way.

        This is the practical payoff of foundation models: **strong results with
        little or no task-specific training data.**

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">13 / 16</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## What to watch out for

        - **Domain gap.** Features learned on everyday photos may transfer poorly
          to very different data (medical scans, satellite imagery).
        - **Compute & memory.** Even *frozen* models can be slow on large data;
          embeddings must be computed once and stored.
        - **Licensing & usage.** Pretrained weights and data come with terms —
          check before you deploy.
        - **Bias.** The pretraining data is not neutral; its biases end up in the
          features.
        - **Evaluate properly.** Always compare against a **baseline** (raw
          pixels, or a simple model) on a **held-out test set** — a fancy
          embedding is not automatically better.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">14 / 16</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Summary

        - A **pretrained model** gives you strong, general **features** for free —
          almost nobody trains vision models from scratch.
        - **Feature extraction** turns each input into an **embedding** you can
          feed to any classifier; **fine-tuning** updates the whole network when
          you have enough data.
        - **Foundation models** (ViT, CLIP, DINO, SAM) push this further, enabling
          **zero- and few-shot** learning.
        - Combined with **PCA / t-SNE** for visualisation and the **supervised
          models** from earlier, pretrained features let us explore and classify
          real image data with very little training.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">15 / 16</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Where to go next

        - **Hands-on:** apply the full pipeline — pretrained embeddings →
          PCA / t-SNE → classifier — to the **course dataset** (details to come).
        - Use the supervised models from earlier in the course (logistic
          regression, forests, a small MLP) on the **embeddings**, and compare
          with the same models on **raw pixels**.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">16 / 16</div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
