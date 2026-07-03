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
# Lecture 0 — Introduction to Machine Learning.
# Run locally with `marimo edit notebooks/00/intro.py`
# or export to WASM for GitHub Pages (see .github/workflows/publish-slides.yml).

import marimo

__generated_with = "0.17.6"
app = marimo.App(
    width="medium",
    layout_file="layouts/intro.slides.json",
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
        # Introduction to Machine Learning

        **Machine Learning with Python**

        EUGLOH \u2014 *Problem Solving Using Open-Source Languages; R and Python*

        [University of Novi Sad](https://www.eugloh.eu/)
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Practical information

        - Slides are Marimo notebooks, exercises are Jupyter notebooks \u2014 all in this repository
        - Clone the repo and follow the setup in the `README.md`
        - Bring a laptop with Python 3.10+; we use [`uv`](https://docs.astral.sh/uv/) for the environment
        - All exercises use `numpy`, `pandas`, `matplotlib`, and `scikit-learn`

        Questions are very welcome \u2014 ask early, ask often.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## What is machine learning?

        > *"A computer program is said to learn from experience E with respect to
        > some class of tasks T and performance measure P, if its performance at
        > tasks in T, as measured by P, improves with experience E."*

        \u2014 Mitchell, 1997
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        In one sentence:

        **Machine learning is the discipline of building systems that learn rules from data, instead of being explicitly programmed.**

        - Traditional programming: *data + program \u2192 output*
        - Machine learning: *data + output \u2192 program*

        The learned *program* (a **model**) is then applied to new data.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Why machine learning?

        - Problems that are **hard to specify** but easy to demonstrate
          - spam detection, image recognition, machine translation
        - Problems that **adapt over time** (recommender systems, fraud detection)
        - Problems at **scale** where hand-written rules break down
        - Problems where **data is abundant** but theory is incomplete

        ML is not magic \u2014 it is a tool that shines when the data is right and the
        question is well-posed.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Types of machine learning

        1. **Supervised learning** \u2014 learn a mapping from inputs to known labels
        2. **Unsupervised learning** \u2014 discover structure in unlabeled data
        3. **Reinforcement learning** \u2014 learn a policy by interacting with an environment

        This course focuses almost entirely on **supervised learning**.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Supervised learning

        - Given a dataset of (input, target) pairs $(x_i, y_i)$
        - Learn a function $f(x) \approx y$
        - Two main flavours:
          - **Regression** \u2014 $y$ is a continuous number (e.g. price, temperature)
          - **Classification** \u2014 $y$ is a discrete label (e.g. spam / not spam)

        Algorithms we will cover: **linear regression, logistic regression, decision trees, random forests**.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Unsupervised learning

        - Only inputs, no labels
        - Goal: discover structure
          - **Clustering** \u2014 group similar points (e.g. customer segments)
          - **Dimensionality reduction** \u2014 find a compact representation

        ### Reinforcement learning

        - An **agent** takes actions in an **environment** to maximise a **reward**
        - Examples: game-playing AIs, robotics, recommender systems

        We will not cover RL in this course.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The ML workflow

        1. **Frame the problem** \u2014 what is the task, what is a good outcome?
        2. **Collect & explore the data** \u2014 EDA, plots, summary statistics
        3. **Preprocess** \u2014 handle missing values, encode categories, scale features
        4. **Split the data** \u2014 training, validation, test
        5. **Choose & train a model** \u2014 fit to the training set
        6. **Evaluate** \u2014 measure performance on held-out data
        7. **Iterate** \u2014 try other models, features, hyperparameters

        Most of the time is spent on **steps 2 and 3**.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## A first example: linear regression

        Let us generate a tiny dataset, fit a linear model, and look at the result.
        """
    )
    return


@app.cell
def _(mo, np, plt):
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 25)
    y = 2.0 * x + 1.0 + rng.normal(scale=2.0, size=x.shape)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, color="#2563eb")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Toy regression dataset")
    mo.as_html(fig)
    return ax, fig, rng, x, y


@app.cell
def _(mo, np, plt, x, y):
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)

    print(f"Learned slope:     {model.coef_[0]:.3f}")
    print(f"Learned intercept: {model.intercept_:.3f}")

    xs = np.linspace(0, 10, 100).reshape(-1, 1)
    ys = model.predict(xs)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, color="#2563eb", label="data")
    ax.plot(xs, ys, color="#dc2626", linewidth=2, label="fit")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Linear regression fit")
    ax.legend()
    mo.as_html(fig)
    return LinearRegression, ax, fig, model, xs, ys


@app.cell
def _(mo):
    mo.md(
        r"""
        ## What just happened?

        - `LinearRegression().fit(X, y)` found the line that **minimises the mean squared error** between the predicted and true values.
        - That is the **entirety of the "learning" step** in classical ML: a numerical optimisation problem.
        - Everything else \u2014 deep learning, transformers, diffusion models \u2014 is built from the same idea: define a model, choose a loss, optimise.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Course outline

        - **Lecture 0 (today):** Introduction to machine learning
        - **Lecture 1:** Linear and logistic regression
        - **Lecture 2:** Decision trees and random forests
        - **Final project:** end-to-end ML pipeline on a dataset of your choice
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Where to go next

        - Lecture 1: linear and logistic regression
        - In the meantime: clone the repo, run `uv sync`, and open the four exercise notebooks in Jupyter Lab
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    return np, plt


if __name__ == "__main__":
    app.run()
