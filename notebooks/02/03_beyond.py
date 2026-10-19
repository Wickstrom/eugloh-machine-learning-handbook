# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "scikit-learn",
#     "scipy",
# ]
# ///
#
# Lecture 2 (Oct 20), Session 3 — Beyond the MLP: architectures that exploit
# structure. Why a flat MLP is wrong for pixels and sequences; convolutional
# networks (locality, weight sharing, pooling, a CNN architecture); transformers
# (self-attention, multi-head attention, positional encoding, the block); and
# autoencoders (representation learning without labels).
# Run locally with `marimo edit notebooks/02/03_beyond.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).
#
# NOTE on scoping: Marimo requires each global name to be owned by exactly
# one cell. All cell-locals are underscore-prefixed. UI elements are created
# in one cell (which shows only the intro text) and *read* in the following
# cell, which displays the widget together with its figure via mo.vstack —
# this keeps every interactive demo on a single slide.

import marimo

__generated_with = "0.17.6"
app = marimo.App(
    width="medium",
    layout_file="layouts/03_beyond.slides.json",
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
        # Beyond the MLP — CNNs & Transformers

        **Machine Learning with Python** — Lecture 2, Session 3 (Oct 20)

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

        **Beyond the MLP** — architectures that exploit the *structure* of data

        - **From vectors to structure** — why an MLP is the wrong tool for
          pixels and sequences
        - **Convolutional networks** — locality, weight sharing, pooling
        - **Transformers** — self-attention, multi-head attention and
          positional encoding
        - **Autoencoders** — representation learning without labels

        and when to reach for **which** model.

        Session 3 of 4 today.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">2 / 16</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Part 1 — Beyond the MLP

        ## From vectors to structure

        An MLP treats its input as a **flat vector** — every neuron sees
        **every** input. For structured data that is a poor fit:

        - **Images.** A 256×256 RGB image is 196,608 numbers; one dense layer
          of 1,000 units already has **~200 million** weights — and it must
          *re-learn* from scratch that a cat is a cat one pixel to the left.
        - **Sequences.** Text and time series are **variable length** and full
          of **long-range** dependencies; a fixed-size vector cannot express
          order.

        The fix is to build the right **inductive bias** into the architecture:
        assume what is true of the data — that pixels are **local**, that
        objects are **translation-invariant**, that tokens are **relational**.
        Two architectures dominate: **CNNs** and **transformers**.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 16</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Convolutional neural networks

        A **convolution** slides a small **kernel** (say 3×3) across the image
        and computes a weighted sum at every position — a **feature map**:

        - **Locality** — each output looks at a small neighbourhood, just like
          a neuron, but the *same* kernel is reused everywhere.
        - **Weight sharing / translation invariance** — one kernel detects a
          feature wherever it appears in the image.
        - **Few parameters** — a 3×3 kernel has 9 weights, no matter the image
          size.

        Stack convolutions and you get a **feature hierarchy**: early layers
        find **edges**, deeper layers compose them into **textures, parts and
        objects**. Below: a digit, an edge-detecting kernel, and its feature map.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 16</div>
        """
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np
    from scipy.signal import convolve2d as _conv
    from sklearn.datasets import load_digits as _load_digits

    _digits = _load_digits()
    _img = _digits.images[0]
    _kernel = _np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=float)
    _edges = _np.abs(_conv(_img, _kernel, mode="same"))

    _fig, _axes = _plt.subplots(1, 3, figsize=(10, 3.6))
    _axes[0].imshow(_img, cmap="gray_r")
    _axes[0].set_title("input digit")
    _axes[1].imshow(_kernel, cmap="RdBu", vmin=-1, vmax=1)
    _axes[1].set_title("3×3 kernel")
    _axes[2].imshow(_edges, cmap="gray_r")
    _axes[2].set_title("feature map (|edges|)")
    for _ax in _axes:
        _ax.set_xticks([])
        _ax.set_yticks([])
    _fig.tight_layout()
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.image(_buf, width="820px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 16</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.patches as _mpatches
    import matplotlib.pyplot as _plt

    _fig, _ax = _plt.subplots(figsize=(11, 2.6))
    _ax.axis("off")
    _ax.set_xlim(0, 1)
    _ax.set_ylim(0, 1)

    _boxes = [
        ("input\n28×28", "#e2e8f0"),
        ("conv 3×3\n32 maps", "#93c5fd"),
        ("pool\n2×2", "#bfdbfe"),
        ("conv 3×3\n64 maps", "#93c5fd"),
        ("pool\n2×2", "#bfdbfe"),
        ("flatten", "#e2e8f0"),
        ("dense\n128", "#86efac"),
        ("softmax\n10", "#fca5a5"),
    ]
    _n = len(_boxes)
    _w = 0.09
    _gap = (1.0 - _n * _w) / (_n + 1)
    _y = 0.32
    _h = 0.36
    for _i, (_label, _c) in enumerate(_boxes):
        _x = _gap + _i * (_w + _gap)
        _ax.add_patch(
            _mpatches.FancyBboxPatch(
                (_x, _y), _w, _h, boxstyle="round,pad=0.01",
                fc=_c, ec="#334155", lw=1.0, zorder=3,
            )
        )
        _ax.text(_x + _w / 2, _y + _h / 2, _label, ha="center", va="center",
                 fontsize=8, zorder=4)
        if _i < _n - 1:
            _ax.annotate(
                "", xy=(_x + _w + _gap * 0.9, _y + _h / 2),
                xytext=(_x + _w + _gap * 0.1, _y + _h / 2),
                arrowprops=dict(arrowstyle="-|>", color="#6b7280", lw=1.3),
            )
    _ax.text(0.5, 0.86, "feature extraction  (conv → ReLU → pool, repeated)",
             ha="center", fontsize=10, color="#111827")
    _ax.text(0.5, 0.08, "classifier head  (dense layers)",
             ha="center", fontsize=10, color="#111827")
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)

    _text = mo.md(
        r"""
        ## A CNN architecture

        A typical CNN repeats one **block** — **conv → ReLU → pool** — and ends
        with a small **classifier head**:

        - **Pooling** (e.g. 2×2 max) **downsamples** the feature maps: the
          spatial size shrinks, the **receptive field** grows (how much of the
          image each unit "sees"), and a little translation tolerance is added.
        - After a few blocks the maps are **flattened** and fed to dense
          layers — from there it is an ordinary MLP.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">6 / 16</div>
        """
    )
    mo.vstack([_text, mo.image(_buf, width="900px")])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Training a CNN

        Nothing about the *learning* changes: it is still **backpropagation**
        and **gradient descent**. The kernels are **learned** — we never
        hand-write an edge detector; the network discovers whatever filters
        minimise the loss.

        This is the punchline of the whole lecture: in Lecture 1 we
        *engineered* features by hand. A CNN **learns** them — and the deeper
        the network, the more abstract the features become.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">7 / 16</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Part 2 — Transformers

        ## Why sequences need attention

        A CNN assumes **locality**; but in a sentence, the word that resolves
        *"it"* may be many tokens away. Earlier models (RNNs) read the sequence
        one step at a time — **serial**, slow, and forgetful over long ranges.

        The **transformer** (Vaswani et al., *Attention is all you need*, 2017)
        throws out recurrence: every token can look at **every other token at
        once**, in parallel. The mechanism is **self-attention**.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">8 / 16</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Self-attention — queries, keys, values

        Each token is first turned into a vector (an **embedding**). For every
        token we then compute three projections:

        - a **query** $q$ — *"what am I looking for?"*,
        - a **key** $k$ — *"what do I offer?"*,
        - a **value** $v$ — *"what do I pass on if attended to?"*.

        The attention weight of token $i$ on token $j$ is the **similarity**
        of $q_i$ and $k_j$, turned into a distribution with softmax:

        $$\alpha_{ij} = \mathrm{softmax}_j\!\left(\frac{q_i \cdot k_j}{\sqrt{d}}\right),
        \qquad \text{out}_i = \sum_j \alpha_{ij}\, v_j$$

        Each token's output is a **weighted mixture** of the values of the
        tokens it deems relevant. The next slide computes this by hand.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">9 / 16</div>
        """
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np

    _tokens = ["the", "cat", "sat"]
    _rng = _np.random.default_rng(2)
    _X = _rng.normal(size=(3, 4))          # 3 tokens, 4-dim embeddings
    _Wq = _rng.normal(size=(4, 4))
    _Wk = _rng.normal(size=(4, 4))
    _Wv = _rng.normal(size=(4, 4))

    _Q, _K, _V = _X @ _Wq, _X @ _Wk, _X @ _Wv
    _scores = _Q @ _K.T / _np.sqrt(4)
    _attn = _np.exp(_scores - _scores.max(axis=1, keepdims=True))
    _attn = _attn / _attn.sum(axis=1, keepdims=True)   # softmax rows

    _fig, _ax = _plt.subplots(figsize=(6.0, 4.8))
    _im = _ax.imshow(_attn, cmap="Blues", vmin=0, vmax=1)
    _ax.set_xticks(range(3), _tokens)
    _ax.set_yticks(range(3), _tokens)
    _ax.set_xlabel("key  (attended to)")
    _ax.set_ylabel("query  (attending from)")
    for _i in range(3):
        for _j in range(3):
            _ax.text(_j, _i, f"{_attn[_i, _j]:.2f}", ha="center", va="center",
                     color="white" if _attn[_i, _j] > 0.5 else "#111827", fontsize=11)
    _ax.set_title("attention weights  $\\alpha_{ij}$  (each row sums to 1)")
    _fig.colorbar(_im, ax=_ax, fraction=0.046, pad=0.04)
    _fig.tight_layout()
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.image(_buf, width="560px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">10 / 16</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Multi-head attention & positional encoding

        - **Multi-head attention** runs several attention "heads" in parallel,
          each with its own $W_q, W_k, W_v$. Different heads track different
          relationships (syntax, coreference, position); their outputs are
          concatenated.
        - Attention is **permutation-invariant** — it has no built-in notion of
          order. **Positional encodings** (added to the embeddings) inject
          position, so *"dog bites man"* ≠ *"man bites dog"*.
        - Stack these blocks over huge datasets and you get **BERT, GPT and
          ViT** — text, and increasingly vision and time series.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">11 / 16</div>
        """
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.patches as _mpatches
    import matplotlib.pyplot as _plt

    _fig, _ax = _plt.subplots(figsize=(5.2, 5.2))
    _ax.axis("off")
    _ax.set_xlim(0, 1)
    _ax.set_ylim(0, 1)

    _steps = [
        ("input embeddings  +  positions", "#e2e8f0"),
        ("multi-head self-attention", "#93c5fd"),
        ("add & norm   (residual)", "#bfdbfe"),
        ("feed-forward network", "#86efac"),
        ("add & norm   (residual)", "#bfdbfe"),
        ("output  →  next block", "#e2e8f0"),
    ]
    _h = 0.115
    _gap = 0.028
    _x, _w = 0.10, 0.62
    _ys = []
    for _i, (_label, _c) in enumerate(_steps):
        _y = 0.90 - _i * (_h + _gap)
        _ys.append(_y)
        _ax.add_patch(_mpatches.FancyBboxPatch((_x, _y), _w, _h,
                     boxstyle="round,pad=0.01", fc=_c, ec="#334155", lw=1.0,
                     zorder=3))
        _ax.text(_x + _w / 2, _y + _h / 2, _label, ha="center", va="center",
                 fontsize=8.5, zorder=4)
        if _i < len(_steps) - 1:
            _ax.annotate("", xy=(_x + _w / 2, _y - _gap + 0.002),
                         xytext=(_x + _w / 2, _y - 0.002),
                         arrowprops=dict(arrowstyle="-|>", color="#6b7280", lw=1.3))
    # residual skip connections around the two sub-layers
    for _top, _bot in [(_ys[1], _ys[2]), (_ys[3], _ys[4])]:
        _ax.annotate("", xy=(_x + _w + 0.11, _bot + _h / 2),
                     xytext=(_x + _w + 0.11, _top + _h / 2),
                     arrowprops=dict(arrowstyle="-|>", color="#dc2626", lw=1.3))
        _ax.plot([_x + _w, _x + _w + 0.11], [_top + _h / 2, _top + _h / 2],
                 color="#dc2626", lw=1.3)
        _ax.plot([_x + _w + 0.11, _x + _w], [_bot + _h / 2, _bot + _h / 2],
                 color="#dc2626", lw=1.3)
    _ax.text(0.72, 0.5, "residual\n$+$", color="#dc2626", fontsize=9,
             ha="left", va="center")
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)

    _text = mo.md(
        r"""
        ## The transformer block

        One block = **multi-head self-attention** + a small **feed-forward
        network**, each wrapped with a **residual connection** and **layer
        normalisation**:

        - **residual** ($x + f(x)$) lets gradients flow through very deep
          networks — the same trick that made deep CNNs trainable;
        - **layer norm** keeps activations well-scaled;
        - the **feed-forward** layer gives every position its own non-linear
          transform.

        Stack $N$ blocks (dozens in an LLM), train with backpropagation and
        **Adam** — it is the *same* machinery as the MLP, scaled up.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">12 / 16</div>
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
        # Part 3 — Learning without labels

        ## Autoencoders — representation learning

        An **autoencoder** is a network trained to reconstruct its own input
        through a narrow **bottleneck**:

        $$x \;\rightarrow\; \text{encoder} \;\rightarrow\; z \;\rightarrow\;
        \text{decoder} \;\rightarrow\; \hat{x}$$

        Because the bottleneck is small, the network must learn a compact
        **representation** $z$ that keeps the important structure. **No labels
        are needed** — this is **representation learning**. Autoencoders are
        used for compression, denoising and **pretraining** (the bridge to
        foundation models, next session).

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">13 / 16</div>
        """
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.datasets import load_digits as _load_digits
    from sklearn.neural_network import MLPRegressor as _MLPR

    _digits = _load_digits()
    _X = _digits.data / 16.0

    _ae = _MLPR(hidden_layer_sizes=(32, 8, 32), max_iter=400, random_state=0)
    _ae.fit(_X, _X)
    _Xhat = _np.clip(_ae.predict(_X[:8]), 0, 1)

    _fig, _axes = _plt.subplots(2, 8, figsize=(12, 3.4))
    for _i in range(8):
        _axes[0, _i].imshow(_digits.images[_i], cmap="gray_r")
        _axes[1, _i].imshow(_Xhat[_i].reshape(8, 8), cmap="gray_r")
        _axes[0, _i].set_xticks([])
        _axes[0, _i].set_yticks([])
        _axes[1, _i].set_xticks([])
        _axes[1, _i].set_yticks([])
    _axes[0, 0].set_ylabel("original", fontsize=9)
    _axes[1, 0].set_ylabel("reconstructed", fontsize=9)
    _fig.tight_layout()
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.md(
                r"""A 64 → 32 → **8** → 32 → 64 autoencoder on handwritten digits,
                trained with `MLPRegressor` on its own input: 64 pixels squeezed
                through **8** numbers and reconstructed well enough to recognise."""
            ),
            mo.image(_buf, width="900px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">14 / 16</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The three model families of this course

        | | logistic regression | trees / forests | neural networks |
        |---|---|---|---|
        | boundary | straight line | axis-aligned steps | smooth, arbitrary |
        | interpretability | high (weights) | tree = flowchart | low |
        | feature scaling | recommended | not needed | required |
        | training cost | seconds | seconds–minutes | minutes–hours |
        | best at | fast baselines, calibrated probabilities | tabular data | images, text, signals |

        Rule of thumb: start with **logistic regression**, try a **random
        forest**, reach for a **neural network** when the data has *structure*
        (pixels, sequences) or the patterns are truly smooth — and reach for
        **CNNs / transformers** when that structure is rich.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">15 / 16</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Summary

        - A flat **MLP ignores structure**; **CNNs** bake in locality, weight
          sharing and pooling — and learn their own features.
        - **Transformers** replace recurrence with **self-attention**: every
          token attends to every other, in parallel; **multi-head** attention
          and **positional encodings** complete the picture.
        - Both are trained with ordinary **backpropagation + Adam** — the
          machinery from Session 2, just stacked deep.
        - **Autoencoders** learn representations **without labels**.

        Next session: **foundation models** — what happens when we train these
        architectures on internet-scale data.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">16 / 16</div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
