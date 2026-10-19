# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "scikit-learn",
# ]
# ///
#
# Lecture 2 (Oct 20), Session 4 — Foundation models. From task-specific models
# to pretrain-then-adapt generalists: the arc through text (embeddings → BERT /
# GPT), vision (AlexNet → self-supervised → ViT), vision-language (CLIP) and
# multimodal generative models — and why this enables applications in
# low-data regimes. Conceptual, figure-driven; no heavy live demos.
# Run locally with `marimo edit notebooks/02/04_foundation_models.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).
#
# NOTE on scoping: Marimo requires each global name to be owned by exactly
# one cell. All cell-locals are underscore-prefixed.

import marimo

__generated_with = "0.17.6"
app = marimo.App(
    width="medium",
    layout_file="layouts/04_foundation_models.slides.json",
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
        # Foundation Models

        **Machine Learning with Python** — Lecture 2, Session 4 (Oct 20)

        EUGLOH — *Problem Solving Using Open-Source Languages; R and Python*

        University of Novi Sad

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">1 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Today's session

        **A different way to build models** — pretrain once, adapt everywhere

        - **From task-specific to foundation models** — what changes
        - **Text** — embeddings → BERT → GPT and in-context learning
        - **Vision** — AlexNet → self-supervised pretraining → ViT
        - **Vision–language** — CLIP, multimodal and generative models
        - **Why it matters for little data** — fine-tuning, linear probes,
          zero/few-shot, LoRA and RAG
        - **Applications, and the caveats**

        Session 4 of 4 today — this wraps the lecture series.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">2 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Part 1 — A different way to build models

        ## From task-specific models to foundation models

        For most of this course we trained **one model per task**: collect
        labels, train, deploy. A **foundation model** turns this around:

        1. **Pretrain** one huge model on **massive, mostly unlabelled** data
           (the whole web, billions of images) to learn general structure.
        2. **Adapt** it to your task with **little or no labelled data** —
           fine-tune it, or just **prompt** it.

        The pretraining objective is usually **self-supervised** (predict the
        next word; decide whether two views match), so it needs **no labels at
        all**. The payoff is **transfer**: a single generalist backbone serves
        thousands of downstream tasks.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Text — from embeddings to pretraining

        - **Word embeddings** (*word2vec*, 2013) learn a vector per word so
          similar words land close together — *king − man + woman ≈ queen*.
          But one vector per word cannot handle **context** (*"bank"*).
        - **Contextual embeddings** (**BERT**, 2018): a transformer reads the
          whole sentence, so each word's vector depends on its **context**.
          BERT is pretrained by **masking** random words and predicting them —
          pure self-supervision.
        - Fine-tune BERT on a task with a **small** labelled set and it beats
          models trained from scratch on far more data.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Text — generative LLMs

        - **GPT** (2018–) is pretrained with a single objective:
          **predict the next token**. At enough scale, that one objective
          yields models that summarise, translate, answer questions and write
          code.
        - **Scaling laws** — performance improves smoothly with model size,
          data and compute; the empirical engine of the last decade.
        - **In-context learning** — a large model can perform a *new* task from
          a few examples **in the prompt**, with **no weight updates**
          (*few-shot prompting*). **Instruction tuning / RLHF** then align it
          to be helpful.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    _text = mo.md(
        r"""
        ## Example — in-context learning

        The headline discovery of **GPT-3** (Brown et al., 2020): a large enough
        language model can perform a **new** task from a few examples placed in
        the **prompt**, with **no gradient updates** at all.

        - a natural-language instruction plus examples raises accuracy;
        - the ability **grows with scale** — the 175B model uses the context far
          better than the 1.3B one;
        - "learning" happens at **inference time**, inside the context window.

        *(Brown et al., Language Models are Few-Shot Learners, 2020 —
        arXiv:2005.14165)*

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">6 / 20</div>
        """
    )
    mo.hstack(
        [
            _text,
            mo.image(
                str(mo.notebook_location() / "public" / "gpt3_in_context.png"),
                width="360px",
            ),
        ],
        widths=[1, 1],
        align="start",
        gap=1.5,
    )
    return


@app.cell
def _(mo):
    _text = mo.md(
        r"""
        ## Example — scaling laws

        Why does scale work? **Kaplan et al. (2020)** measured test loss across
        hundreds of runs and found it falls as a smooth **power law** in each of
        three factors:

        - **compute**, **dataset size** and **parameters**;
        - the three must grow *together* — scaling one alone quickly saturates;
        - this empirical regularity is why the field bets on bigger models and
          bigger data.

        *(Kaplan et al., Scaling Laws for Neural Language Models, 2020 —
        arXiv:2001.08361)*

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">7 / 20</div>
        """
    )
    mo.hstack(
        [
            _text,
            mo.image(
                str(mo.notebook_location() / "public" / "scaling_laws.png"),
                width="360px",
            ),
        ],
        widths=[1, 1],
        align="start",
        gap=1.5,
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt

    _events = [
        (2012, "AlexNet"),
        (2013, "word2vec"),
        (2017, "Transformer"),
        (2018, "BERT / GPT"),
        (2020, "ViT / GPT-3"),
        (2021, "CLIP / Diffusion"),
        (2022, "ChatGPT"),
        (2024, "Multimodal LLMs"),
    ]
    _fig, _ax = _plt.subplots(figsize=(11, 2.8))
    _ax.axhline(0, color="#94a3b8", lw=2, zorder=1)
    for _i, (_year, _label) in enumerate(_events):
        _up = _i % 2 == 0
        _ax.plot([_year], [0], "o", color="#2563eb", ms=8, zorder=3)
        _ax.annotate(
            f"{_label}\n{_year}",
            xy=(_year, 0),
            xytext=(_year, 0.35 if _up else -0.35),
            ha="center", va="center", fontsize=9,
            arrowprops=dict(arrowstyle="-", color="#94a3b8", lw=1),
        )
    _ax.set_xlim(2010.5, 2025.5)
    _ax.set_ylim(-0.8, 0.8)
    _ax.axis("off")
    _ax.set_title("From task-specific models to foundation models")
    _fig.tight_layout()
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.image(_buf, width="940px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">8 / 20</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Vision — from supervised to self-supervised

        - **AlexNet** (2012) won ImageNet with a deep **CNN** and GPUs — but it
          was supervised and needed **1.2 million** labelled images.
        - The field then asked: can we pretrain **without** labels? Yes:
          - **contrastive learning** (SimCLR, MoCo) — pull two augmentations of
            the *same* image together, push different images apart;
          - **masked image modelling** (MAE) — hide patches and reconstruct
            them, the vision analogue of BERT.
        - These general visual features **transfer** to detection, segmentation
          and medical imaging — often with only a handful of labels.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">9 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.patches as _mpatches
    import matplotlib.pyplot as _plt
    from sklearn.datasets import load_digits as _load_digits

    _img = _load_digits().images[0]
    _fig, _ax = _plt.subplots(figsize=(5.4, 4.6))
    _ax.imshow(_img, cmap="gray_r", extent=[0, 8, 0, 8])
    for _g in range(0, 9, 2):
        _ax.plot([0, 8], [_g, _g], color="#2563eb", lw=1.2, alpha=0.8)
        _ax.plot([_g, _g], [0, 8], color="#2563eb", lw=1.2, alpha=0.8)
    _ax.add_patch(_mpatches.Rectangle((4, 4), 2, 2, fill=False, ec="#dc2626", lw=2.2))
    _ax.text(5, 5, "patch", color="#dc2626", fontsize=10, ha="center", va="center",
             fontweight="bold")
    _ax.set_xticks([])
    _ax.set_yticks([])
    _ax.set_title("an image = a sequence of patches (tokens)")
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)

    _text = mo.md(
        r"""
        ## Vision — Vision Transformers (ViT)

        The transformer is **not text-specific**. **ViT** (2020) chops an image
        into **patches** (e.g. 16×16), embeds each patch as a **token**, and
        runs a plain transformer over them:

        - patches play the role of words; **positional encodings** keep their
          layout;
        - with enough pretraining data, ViT matches or beats CNNs — and scales
          further;
        - this **unifies** vision and language under one architecture — the key
          to the multimodal models next.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">10 / 20</div>
        """
    )
    mo.hstack(
        [_text, mo.image(_buf, width="430px")],
        widths=[1, 1], align="start", gap=1.5,
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Part 2 — Vision and language together

        ## CLIP — connecting images and text

        **CLIP** (2021) trains **two encoders** — one for images, one for text
        — on **400 million** image–caption pairs:

        - for each batch, maximise the similarity of the **correct**
          (image, caption) pairs and minimise it for every mismatch
          (**contrastive** loss);
        - the result is a **shared embedding space** where an image and its
          description land in the same place.

        Once trained, CLIP classifies images **zero-shot**: embed the class
        names ("a photo of a cat"), embed the image, pick the closest — with
        **no training on the target classes at all**.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">11 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np

    _S = _np.full((4, 4), 0.15)
    _np.fill_diagonal(_S, 1.0)
    _fig, _ax = _plt.subplots(figsize=(6.2, 5.0))
    _im = _ax.imshow(_S, cmap="Greens", vmin=0, vmax=1)
    _ax.set_xticks(range(4), [f"text {_i + 1}" for _i in range(4)])
    _ax.set_yticks(range(4), [f"image {_i + 1}" for _i in range(4)])
    _ax.set_xlabel("text embeddings")
    _ax.set_ylabel("image embeddings")
    for _i in range(4):
        for _j in range(4):
            _ax.text(_j, _i, "✓" if _i == _j else "·", ha="center", va="center",
                     color="#111827", fontsize=12)
    _ax.set_title("contrastive loss pushes the diagonal up")
    _fig.tight_layout()
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)

    _text = mo.md(
        r"""
        ## The CLIP idea, in one picture

        An **image encoder** and a **text encoder** map their inputs into a
        *shared* space; the training signal is the **similarity matrix**:

        - the **diagonal** (matched image–caption pairs) is pulled together;
        - the **off-diagonal** (mismatches) is pushed apart.

        Because both live in one space, you can compare an image to *arbitrary
        text* — the basis for **zero-shot classification**, retrieval and the
        vision towers inside multimodal LLMs.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">12 / 20</div>
        """
    )
    mo.hstack(
        [_text, mo.image(_buf, width="430px")],
        widths=[1, 1], align="start", gap=1.5,
    )
    return


@app.cell
def _(mo):
    _text = mo.md(
        r"""
        ## Example — CLIP, from the paper

        The actual diagram (Radford et al., 2021) — the same three ideas as the
        previous slide, drawn out:

        - **(1) contrastive pre-training** — match $N$ images with $N$ captions
          in a batch;
        - **(2) build a classifier from text** — embed the class names;
        - **(3) zero-shot prediction** — pick the caption whose embedding is
          *closest* to the image. No labelled images for the target task.

        *(Radford et al., Learning Transferable Visual Models From Natural
        Language Supervision, 2021 — arXiv:2103.00020)*

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">13 / 20</div>
        """
    )
    mo.hstack(
        [
            _text,
            mo.image(
                str(mo.notebook_location() / "public" / "clip_overview.png"),
                width="360px",
            ),
        ],
        widths=[1, 1],
        align="start",
        gap=1.5,
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Multimodal & generative models

        - **Vision–language**: captioning, visual question answering, and
          **multimodal LLMs** (GPT-4V, LLaVA) that both *see* and *talk*.
        - **Generative vision**: **diffusion models** (Stable Diffusion,
          DALL·E) learn to denoise random noise into an image under a text
          prompt — the same foundation idea, used to **generate** rather than
          classify.
        - **One recipe, many modalities**: audio (Whisper), video, protein
          sequences, even tabular data — a shared encoder plus a task head,
          pretrained at scale.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">14 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Part 3 — Why this matters for little data

        ## Data efficiency is the whole point

        The headline benefit is that foundation models turn a **data** problem
        into an **adaptation** problem:

        - **Pretrain → fine-tune** — the backbone already "knows" general
          structure; a little labelled data adapts it to your task.
        - **Linear probe** — freeze the backbone and fit a single linear layer
          on top; often the strongest baseline with **very few** labels.
        - **Zero / few-shot** — with CLIP or an LLM you may need **no training
          at all** — just describe the task.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">15 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np

    _labels = _np.array([1, 2, 4, 8, 16, 32, 64, 128])
    _scratch = 0.5 + 0.45 * (1 - _np.exp(-_labels / 40.0))
    _foundation = 0.80 + 0.18 * (1 - _np.exp(-_labels / 6.0))

    _fig, _ax = _plt.subplots(figsize=(6.6, 4.4))
    _ax.semilogx(_labels, _scratch, "o-", color="#dc2626",
                 label="trained from scratch")
    _ax.semilogx(_labels, _foundation, "o-", color="#16a34a",
                 label="fine-tuned foundation model")
    _ax.set_xlabel("number of labelled examples (log scale)")
    _ax.set_ylabel("test accuracy")
    _ax.set_ylim(0.4, 1.02)
    _ax.set_title("Schematic — foundation models need far fewer labels")
    _ax.legend(fontsize=8, loc="lower right")
    _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)

    _text = mo.md(
        r"""
        ## Few labels, high accuracy

        *(Illustrative curves — the shape, not the exact numbers, is the point.)*

        - A model trained **from scratch** needs many labels to climb.
        - A pretrained backbone adapted to the task is already good with
          **a handful** of examples, and improves fast.

        This is exactly what makes foundation models usable in the
        **small-data regimes** that dominate real projects — rare medical
        conditions, niche industrial defects, small-company datasets.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">16 / 20</div>
        """
    )
    mo.hstack(
        [_text, mo.image(_buf, width="450px")],
        widths=[1, 1], align="start", gap=1.5,
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Adapting cheaply

        Fully fine-tuning a billion-parameter model is expensive; a few tricks
        make adaptation practical:

        - **Parameter-efficient fine-tuning (PEFT)** — **LoRA** / adapters
          *freeze* the backbone and train small injected matrices: fractions of
          a percent of the parameters.
        - **Prompt engineering** — the task lives in the prompt, not the
          weights.
        - **Retrieval-augmented generation (RAG)** — fetch relevant documents
          and put them in the context, so the model answers from **your** data
          *without* retraining.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">17 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Applications

        - **Language** — chatbots, translation, summarisation, code assistants.
        - **Vision** — zero-shot classification, segmentation (SAM), detection.
        - **Vision–language** — captioning, visual QA, document understanding,
          medical-imaging reports.
        - **Science** — protein structure (AlphaFold), drug discovery, remote
          sensing.
        - **The low-data cases that used to be hopeless** — rare diseases,
          niche defects on a factory line, small tabular datasets.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">18 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Limitations & considerations

        - **Compute & cost** — pretraining takes millions of GPU-hours;
          fine-tuning still needs hardware.
        - **Data, licensing & privacy** — scraped data raises consent and
          copyright questions — and never put private data into a prompt.
        - **Bias** — models inherit and can amplify biases in their training
          data.
        - **Hallucination** — fluent text is not the same as fact; verify
          outputs.
        - **Evaluation & reproducibility** — benchmarks saturate and models
          change; the classical workflow (validation sets, honest baselines)
          still matters.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">19 / 20</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Summary

        - **Foundation models** are **pretrained** (self-supervised, at scale)
          and then **adapted** — one generalist backbone, many tasks.
        - The arc: **text** (embeddings → BERT/GPT) → **vision** (AlexNet →
          self-supervised → ViT) → **vision–language** (CLIP) → **multimodal**.
        - Their real superpower is **data efficiency** — zero/few-shot, and
          cheap adaptation via **LoRA**, prompting and **RAG**.
        - The training machinery is the same as Session 2 — **backprop + Adam**
          — just at a scale that changes what is possible.
        - Back to reality: for small **labelled** problems, logistic regression
          and random forests still win on **cost and interpretability**.

        That wraps the lecture series — thanks!

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">20 / 20</div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
