from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from .data import FEATURES, TARGET


def compute_eda(
    raw_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, float]]:
    df = pd.read_csv(raw_path)

    x = df[FEATURES].to_numpy()
    y = df[TARGET].to_numpy()

    linear = LogisticRegression(
        max_iter=2000,
        random_state=42,
    ).fit(x, y)

    pred = linear.predict(x)

    summary = {
        "rows": float(len(df)),
        "columns": float(len(df.columns)),
        "features": float(len(FEATURES)),
        "missing_cells": float(
            df.isna().sum().sum()
        ),
        "duplicate_rows": float(
            df.duplicated().sum()
        ),
        "class_0_share": float(
            (y == 0).mean()
        ),
        "class_1_share": float(
            (y == 1).mean()
        ),
        "x1_mean": float(df["x1"].mean()),
        "x1_std": float(df["x1"].std()),
        "x1_min": float(df["x1"].min()),
        "x1_max": float(df["x1"].max()),
        "x2_mean": float(df["x2"].mean()),
        "x2_std": float(df["x2"].std()),
        "x2_min": float(df["x2"].min()),
        "x2_max": float(df["x2"].max()),
        "x1_x2_corr": float(
            df[["x1", "x2"]]
            .corr()
            .iloc[0, 1]
        ),
        "linear_train_accuracy": float(
            accuracy_score(y, pred)
        ),
    }

    stats = pd.DataFrame(
        [
            {
                "metric": key,
                "value": value,
            }
            for key, value in summary.items()
        ]
    )

    return stats, summary


def plot_problem(
    raw_path: str | Path,
    output_path: str | Path,
    seed: int = 42,
) -> None:
    df = pd.read_csv(raw_path)

    rng = np.random.default_rng(seed)

    if len(df) > 5000:
        idx = rng.choice(
            len(df),
            size=5000,
            replace=False,
        )
        draw = df.iloc[idx]
    else:
        draw = df

    x = df[FEATURES].to_numpy()
    y = df[TARGET].to_numpy()

    model = LogisticRegression(
        max_iter=2000,
        random_state=seed,
    ).fit(x, y)

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    for label in [0, 1]:
        part = draw[
            draw[TARGET] == label
        ]

        ax.scatter(
            part["x1"],
            part["x2"],
            s=8,
            alpha=0.45,
            label=f"class {label}",
        )

    xs = np.linspace(
        df["x1"].min(),
        df["x1"].max(),
        200,
    )

    coef = model.coef_[0]
    intercept = model.intercept_[0]

    if abs(coef[1]) > 1e-12:
        ys = -(
            coef[0] * xs + intercept
        ) / coef[1]

        ax.plot(
            xs,
            ys,
            linewidth=2,
            label="linear boundary",
        )

    acc = accuracy_score(
        y,
        model.predict(x),
    )

    ax.set_title(
        "make_moons: нелинейность; "
        f"accuracy линейной границы = {acc:.4f}"
    )
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()
    ax.grid(alpha=0.2)

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=150,
    )
    plt.close(fig)


def plot_confusion(
    raw_path: str | Path,
    output_path: str | Path,
) -> None:
    df = pd.read_csv(raw_path)

    x = df[FEATURES].to_numpy()
    y = df[TARGET].to_numpy()

    model = LogisticRegression(
        max_iter=2000,
        random_state=42,
    ).fit(x, y)

    cm = confusion_matrix(
        y,
        model.predict(x),
    )

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    image = ax.imshow(cm)
    fig.colorbar(
        image,
        ax=ax,
    )

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    ax.set_xlabel(
        "Предсказанный класс"
    )
    ax.set_ylabel(
        "Истинный класс"
    )
    ax.set_title(
        "Линейный ориентир: матрица ошибок"
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=150,
    )
    plt.close(fig)


def plot_class_balance(
    raw_path: str | Path,
    output_path: str | Path,
) -> None:
    df = pd.read_csv(raw_path)

    counts = (
        df[TARGET]
        .value_counts()
        .sort_index()
        .rename_axis("class")
        .reset_index(name="count")
    )

    fig, ax = plt.subplots(
        figsize=(6, 4)
    )

    sns.barplot(
        data=counts,
        x="class",
        y="count",
        ax=ax,
    )

    total = len(df)

    for index, row in counts.iterrows():
        share = row["count"] / total

        ax.text(
            index,
            row["count"],
            f"{int(row['count'])}\n({share:.1%})",
            ha="center",
            va="bottom",
        )

    ax.set_xlabel("Класс")
    ax.set_ylabel(
        "Количество объектов"
    )
    ax.set_title(
        "Баланс классов"
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=150,
    )
    plt.close(fig)


def plot_feature_distributions(
    raw_path: str | Path,
    output_path: str | Path,
) -> None:
    df = pd.read_csv(raw_path)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4),
    )

    for ax, feature in zip(
        axes,
        FEATURES,
    ):
        sns.histplot(
            data=df,
            x=feature,
            hue=TARGET,
            bins=40,
            stat="density",
            common_norm=False,
            element="step",
            ax=ax,
        )

        ax.set_title(
            f"Распределение {feature}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=150,
    )
    plt.close(fig)


def plot_correlation(
    raw_path: str | Path,
    output_path: str | Path,
) -> None:
    df = pd.read_csv(raw_path)

    corr = df[
        ["x1", "x2", "y"]
    ].corr()

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    sns.heatmap(
        corr,
        annot=True,
        fmt=".3f",
        square=True,
        ax=ax,
    )

    ax.set_title(
        "Корреляционная матрица"
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=150,
    )
    plt.close(fig)
