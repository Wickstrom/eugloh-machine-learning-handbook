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
# Friday on-site session (Oct 23) — Lecture 1: Dimensionality reduction.
# Principal component analysis (PCA) and t-distributed stochastic neighbour
# embedding (t-SNE): what they do, how they work, how to read them, and how to
# use them together with the supervised models from earlier in the course.
# Run locally with `marimo edit notebooks/03/01_dimensionality_reduction.py`
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
    layout_file="layouts/01_dimensionality_reduction.slides.json",
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
        # Dimensionality Reduction

        ## PCA & t-SNE

        **Machine Learning with Python** — On-site session (Oct 23)

        EUGLOH — *Problem Solving Using Open-Source Languages; R and Python*

        University of Novi Sad

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">1 / 18</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Today's session

        - **Why** reduce dimensions at all?
        - **PCA** — finding the directions where the data varies most
        - **t-SNE** — drawing a map that keeps neighbours together
        - How to **read** these plots — and how to **avoid fooling yourself**
        - How both fit into the pipeline with the models you already know

        On-site session — the first of two today.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">2 / 18</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Part 1 — Why reduce dimensions?

        Real data has **many features**: a tiny $8\times8$ image is already 64
        numbers, a small image at $224\times224$ is **150 528**. We rarely need
        all of them at once.

        - **Visualisation** — we cannot draw 64 dimensions; project to 2 so our eyes can help.
        - **The curse of dimensionality** — in high dimensions data becomes *sparse* and distances stop being informative.
        - **Compression & denoising** — keep the signal, drop the noise.
        - **Speed** — fewer features mean faster training and prediction.

        The goal: keep the **structure**, throw away the **redundancy**.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">3 / 18</div>
        """
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np

    _rng = _np.random.default_rng(0)
    _dims = [1, 2, 3, 5, 8, 12, 20, 30, 50, 80]
    _ratios = []
    for _d in _dims:
        _X = _rng.random((300, _d))
        _D = _np.sqrt(((_X[:, None, :] - _X[None, :, :]) ** 2).sum(-1))
        _np.fill_diagonal(_D, _np.nan)
        _ratios.append(_np.median(_np.nanmax(_D, axis=1) / _np.nanmin(_D, axis=1)))

    _fig, _ax = _plt.subplots(figsize=(7.4, 4.0))
    _ax.plot(_dims, _ratios, "o-", color="#2563eb", lw=2)
    _ax.axhline(1, color="gray", ls="--", lw=0.8)
    _ax.set_xscale("log")
    _ax.set_xlabel("number of dimensions")
    _ax.set_ylabel("farthest / nearest neighbour distance")
    _ax.set_title("In high dimensions, everything is roughly equidistant")
    _ax.grid(alpha=0.3)
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.image(_buf, width="680px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">4 / 18</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Part 2 — PCA

        ## The idea: follow the variance

        PCA looks for the direction along which the data **varies the most**,
        then the next-most, perpendicular to the first, and so on.

        - Projecting onto those directions keeps as much **spread** as possible.
        - Two correlated features carry mostly the *same* information — PCA
          replaces them with one direction and loses almost nothing.
        - In plain words: **rotate the axes** so they line up with how the data
          is stretched, then drop the squished directions.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">5 / 18</div>
        """
    )
    return


@app.cell
def _(mo):
    angle_slider = mo.ui.slider(
        start=0, stop=180, step=1, value=20,
        label="projection angle (degrees)", show_value=True, debounce=True,
    )
    mo.md(
        r"""
        ## PCA — find the best direction

        The grey cloud below is real data. Drag the angle to choose a direction
        to project onto (orange dots are the projections). The right panel shows
        how much **variance** that direction captures. Which angle wins?

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">6 / 18</div>
        """
    )
    return (angle_slider,)


@app.cell
def _(angle_slider, mo):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np

    _rng = _np.random.default_rng(1)
    _X = _rng.multivariate_normal([0, 0], [[2.2, 1.6], [1.6, 1.4]], size=200)

    _theta = _np.deg2rad(angle_slider.value)
    _u = _np.array([_np.cos(_theta), _np.sin(_theta)])
    _proj = _X @ _u
    _var_now = _proj.var()

    # variance of the projection as a function of angle
    _angles = _np.deg2rad(_np.arange(0, 181))
    _M = _np.stack([_np.cos(_angles), _np.sin(_angles)], axis=1)
    _vars = (_X @ _M.T).var(axis=0)

    # PCA/SVD reference (ground truth)
    _U, _S, _Vt = _np.linalg.svd(_X - _X.mean(0), full_matrices=False)
    _best_dir = _Vt[0]
    _eig = (_S ** 2) / (_X.shape[0] - 1)
    _total_var = _eig.sum()
    _best_angle = _np.rad2deg(_np.arctan2(_best_dir[1], _best_dir[0])) % 180

    _fig, _axes = _plt.subplots(1, 2, figsize=(11, 4.4))

    _t = _np.linspace(-4, 4, 2)
    _axes[0].plot(_t * _u[0], _t * _u[1], color="#f59e0b", lw=2, label="your direction")
    _tb = _np.linspace(-4, 4, 2)
    _axes[0].plot(_tb * _best_dir[0], _tb * _best_dir[1], color="#16a34a", lw=1.5, ls="--", label="best (PCA)")
    _axes[0].scatter(_X[:, 0], _X[:, 1], s=14, color="#94a3b8")
    _axes[0].scatter(_u[0] * _proj, _u[1] * _proj, s=8, color="#f59e0b")
    _axes[0].set_aspect("equal")
    _axes[0].set_title("data and the projection onto one direction")
    _axes[0].legend(fontsize=8)
    _axes[0].grid(alpha=0.3)

    _axes[1].plot(_np.rad2deg(_angles), _vars, color="#94a3b8")
    _axes[1].scatter([angle_slider.value], [_var_now], color="#f59e0b", s=60, zorder=5)
    _axes[1].axvline(_best_angle, color="#16a34a", ls="--", lw=1.5)
    _axes[1].set_xlabel("projection angle (degrees)")
    _axes[1].set_ylabel("variance captured")
    _axes[1].set_title(f"you capture {_var_now / _total_var:.0%} of the variance")
    _axes[1].grid(alpha=0.3)

    _fig.tight_layout()
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            angle_slider,
            mo.image(_buf, width="900px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">7 / 18</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## PCA — the recipe

        1. **Centre** the data: subtract the mean of each feature.
        2. **Find the directions of maximum variance** — the eigenvectors of the
           covariance matrix (equivalently, the **SVD** of the centred data).
        3. **Project**: $Z = X W$, where the columns of $W$ are the top $k$
           directions.
        4. Each direction captures a **variance** (its eigenvalue) — sum them to
           see how much you kept.

        In plain words: **centre, find the stretch directions, keep the big ones.**

        Two things to remember: always **standardise** your features first (PCA
        is scale-sensitive), and the components are **uncorrelated** by
        construction.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">8 / 18</div>
        """
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.datasets import load_digits as _load_digits
    from sklearn.decomposition import PCA as _PCA
    from sklearn.preprocessing import StandardScaler as _Scaler

    _digits = _load_digits()
    _X = _Scaler().fit_transform(_digits.data)
    _y = _digits.target

    _pca = _PCA().fit(_X)
    _cum = _np.cumsum(_pca.explained_variance_ratio_)

    _Z = _PCA(n_components=2).fit_transform(_X)

    _fig, _axes = _plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    _axes[0].plot(_np.arange(1, len(_cum) + 1), _cum, color="#2563eb", lw=2)
    _axes[0].axhline(0.9, color="gray", ls="--", lw=0.8)
    _axes[0].axvline(_np.argmax(_cum >= 0.9) + 1, color="#dc2626", ls=":", lw=1.2)
    _axes[0].set_xlabel("number of components")
    _axes[0].set_ylabel("cumulative explained variance")
    _axes[0].set_title(f"digits: {_np.argmax(_cum >= 0.9) + 1} components reach 90%")
    _axes[0].grid(alpha=0.3)

    _sc = _axes[1].scatter(_Z[:, 0], _Z[:, 1], c=_y, cmap="tab10", s=12, vmin=-0.5, vmax=9.5)
    _axes[1].set_xlabel("PC 1")
    _axes[1].set_ylabel("PC 2")
    _axes[1].set_title(f"first two components keep {_cum[1]:.0%} of the variance")
    _axes[1].grid(alpha=0.3)
    _fig.colorbar(_sc, ax=_axes[1], ticks=range(10), label="digit")

    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.image(_buf, width="900px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">9 / 18</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.datasets import load_digits as _load_digits
    from sklearn.decomposition import PCA as _PCA

    _digits = _load_digits()
    _X = _digits.data / 16.0
    _idx = [0, 6, 14, 20, 27]
    _ks = [2, 8, 24, 64]

    _fig, _axes = _plt.subplots(len(_ks) + 1, len(_idx), figsize=(9, 6))
    for _j, _i in enumerate(_idx):
        _axes[0, _j].imshow(_X[_i].reshape(8, 8), cmap="gray_r")
    _axes[0, 0].set_ylabel("original", fontsize=9)
    for _r, _k in enumerate(_ks, start=1):
        _p = _PCA(n_components=_k).fit(_X)
        _Xr = _p.inverse_transform(_p.transform(_X))
        for _j, _i in enumerate(_idx):
            _axes[_r, _j].imshow(_Xr[_i].reshape(8, 8), cmap="gray_r")
        _axes[_r, 0].set_ylabel(f"{_k} comps\n{_p.explained_variance_ratio_.sum():.0%}", fontsize=8)
    for _ax in _axes.ravel():
        _ax.set_xticks([])
        _ax.set_yticks([])
    _fig.suptitle("Reconstruction: compression, and gentle denoising")
    _fig.tight_layout()
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.image(_buf, width="640px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">10 / 18</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## PCA — what to watch out for

        - **It is linear.** Curved or clustered structure is not captured; if the
          data lies on a curve, PCA will flatten it.
        - **Scale matters.** An unscaled feature with a large range dominates.
          Standardise first.
        - **Maximum variance ≠ best separation.** PCA does not look at the
          labels. A direction with little variance can be the one that separates
          the classes.
        - **Components are hard to name.** Each is a mix of all original
          features, so interpretability drops.

        Still the best **first** thing to try: fast, deterministic, and a good
        sanity check before fancier methods.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">11 / 18</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Part 3 — t-SNE

        ## The idea: keep neighbours together

        t-SNE does not care about variance. It asks a **neighbourhood** question:
        *which points are close to which, in the original space?* — then lays
        them out in 2-D so that the same points stay close.

        - High-dimensional similarities become **probabilities** (a Gaussian
          around each point).
        - Low-dimensional similarities do the same with a heavier-tailed
          distribution.
        - It then nudges the points to make the two sets of probabilities match
          (minimising a **KL divergence**).

        **Perplexity** controls how many neighbours each point cares about —
        roughly the "attention span" of the map.

        In plain words: **attract neighbours, push strangers apart.**

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">12 / 18</div>
        """
    )
    return


@app.cell
def _(mo):
    perp_slider = mo.ui.slider(
        start=5, stop=50, step=5, value=30,
        label="perplexity", show_value=True, debounce=True,
    )
    mo.md(
        r"""
        ## t-SNE — the role of perplexity

        The same digits data, mapped to 2-D with different perplexities. Low
        values focus on very local structure (many tiny islands); high values
        blend neighbours together (fewer, larger clumps). There is no single
        "correct" value — look for a map that is **stable across a few
        settings**.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">13 / 18</div>
        """
    )
    return (perp_slider,)


@app.cell
def _(mo, perp_slider):
    import io as _io

    import matplotlib.pyplot as _plt
    import numpy as _np
    from sklearn.datasets import load_digits as _load_digits
    from sklearn.manifold import TSNE as _TSNE
    from sklearn.preprocessing import StandardScaler as _Scaler

    _digits = _load_digits()
    _X = _Scaler().fit_transform(_digits.data)[:1000]
    _y = _digits.target[:1000]

    _Y = _TSNE(
        n_components=2, perplexity=perp_slider.value, init="pca",
        learning_rate="auto", max_iter=500, random_state=0,
    ).fit_transform(_X)

    _fig, _ax = _plt.subplots(figsize=(6.2, 5.2), layout="constrained")
    _sc = _ax.scatter(_Y[:, 0], _Y[:, 1], c=_y, cmap="tab10", s=14, vmin=-0.5, vmax=9.5)
    _ax.set_title(f"t-SNE of digits — perplexity = {perp_slider.value}")
    _ax.set_xticks([])
    _ax.set_yticks([])
    _fig.colorbar(_sc, ax=_ax, ticks=range(10), label="digit")
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            perp_slider,
            mo.image(_buf, width="600px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">14 / 18</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## t-SNE — how to read it (and not fool yourself)

        - **Cluster distances are not meaningful.** Two blobs far apart are not
          "more different" than two blobs close together.
        - **Cluster sizes are not meaningful** either — t-SNE expands dense
          clusters and squeezes sparse ones.
        - **It is stochastic.** Run it a few times; trust what is stable.
        - **Hyperparameters matter.** Perplexity (and the learning rate) change
          the picture; do not over-interpret one run.
        - **Do not feed the 2-D coordinates into a classifier.** It is built for
          *looking*, not for prediction (use PCA or the raw features for that).

        t-SNE is a **microscope for local structure**, not a map of global
        distances.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">15 / 18</div>
        """
    )
    return


@app.cell
def _(mo):
    import io as _io

    import matplotlib.pyplot as _plt
    from sklearn.datasets import load_digits as _load_digits
    from sklearn.decomposition import PCA as _PCA
    from sklearn.manifold import TSNE as _TSNE
    from sklearn.preprocessing import StandardScaler as _Scaler

    _digits = _load_digits()
    _X = _Scaler().fit_transform(_digits.data)[:1000]
    _y = _digits.target[:1000]

    _Zp = _PCA(n_components=2, random_state=0).fit_transform(_X)
    _Zt = _TSNE(
        n_components=2, perplexity=30, init="pca",
        learning_rate="auto", max_iter=500, random_state=0,
    ).fit_transform(_X)

    _fig, _axes = _plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for _ax, _Z, _name in [
        (_axes[0], _Zp, "PCA (linear)"),
        (_axes[1], _Zt, "t-SNE (neighbourhoods)"),
    ]:
        _sc = _ax.scatter(_Z[:, 0], _Z[:, 1], c=_y, cmap="tab10", s=12, vmin=-0.5, vmax=9.5)
        _ax.set_title(_name)
        _ax.set_xticks([])
        _ax.set_yticks([])
    _fig.colorbar(_sc, ax=_axes, ticks=range(10), label="digit", fraction=0.025)
    _fig.suptitle("The same digits, two very different views")
    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
    _plt.close(_fig)
    _buf.seek(0)
    mo.vstack(
        [
            mo.image(_buf, width="900px"),
            mo.md(r"""<div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">16 / 18</div>"""),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Summary

        - **PCA** finds the directions of **maximum variance** by rotating the
          axes; it is **linear, fast and deterministic**, and great for
          compression, denoising and a first look.
        - **t-SNE** builds a 2-D **neighbourhood map**; it is the better tool for
          *seeing* clusters, but its distances and sizes are **not** meaningful
          and it must not be used as input features.
        - **Standardise first**, and remember **variance is not separation**.
        - Choose the tool for the **job**: PCA to reduce/compress/feed a model,
          t-SNE to look.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">17 / 18</div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Where to go next

        - **Next lecture:** **pretrained models** — using powerful feature
          extractors, and how combining them with PCA/t-SNE and the supervised
          models from earlier lets us visualise and classify real image data.
        - We will apply all of this to the **course dataset** in the hands-on
          part.

        <div style="position:fixed;bottom:12px;left:16px;font-size:13px;color:#888;font-family:system-ui,sans-serif;">18 / 18</div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
