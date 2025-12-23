from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

AGGREGATION_METHODS = {
    "sum": "sum",
    "mean": "mean",
    "median": "median",
    "count": "count",
}


def time_series_plot(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    agg_method: str = "sum",
) -> Optional[plt.Figure]:
    if not date_col or not value_col:
        return None
    if date_col not in df.columns or value_col not in df.columns:
        return None

    date_series = pd.to_datetime(df[date_col], errors="coerce")
    numeric_series = pd.to_numeric(df[value_col], errors="coerce")
    valid = df[date_series.notna() & numeric_series.notna()].copy()
    if valid.empty:
        return None
    series = valid.copy()
    series[date_col] = date_series.loc[valid.index]
    series[value_col] = numeric_series.loc[valid.index]
    aggregated = (
        series.groupby(date_col)[value_col]
        .agg(AGGREGATION_METHODS.get(agg_method, "sum"))
        .sort_index()
    )
    if aggregated.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(aggregated.index, aggregated.values, marker="o")
    ax.set_xlabel(date_col)
    ax.set_ylabel(value_col)
    ax.set_title(f"{value_col} over time ({agg_method})")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    return fig


def distribution_plot(
    df: pd.DataFrame,
    column: str,
    plot_type: str = "histogram",
) -> Optional[plt.Figure]:
    if not column or column not in df.columns:
        return None
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    if plot_type == "box":
        ax.boxplot(series, vert=True)
        ax.set_ylabel(column)
        ax.set_title(f"{column} distribution (box plot)")
    else:
        ax.hist(series, bins=30, color="#1f77b4", alpha=0.7)
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        ax.set_title(f"{column} distribution (histogram)")
    ax.grid(True, alpha=0.3)
    return fig


def category_plot(
    df: pd.DataFrame,
    category_col: str,
    value_col: Optional[str],
    agg_method: str = "sum",
    chart_type: str = "bar",
) -> Optional[plt.Figure]:
    if not category_col or category_col not in df.columns:
        return None
    data = df.dropna(subset=[category_col])
    if data.empty:
        return None
    if value_col and value_col in df.columns and value_col != category_col:
        numeric_series = pd.to_numeric(data[value_col], errors="coerce")
        valid = data[numeric_series.notna()].copy()
        if valid.empty:
            return None
        valid[value_col] = numeric_series.loc[valid.index]
        grouped = (
            valid.groupby(category_col)[value_col]
            .agg(AGGREGATION_METHODS.get(agg_method, "sum"))
            .sort_values(ascending=False)
        )
        title = f"{value_col} by {category_col} ({agg_method})"
    else:
        grouped = data[category_col].value_counts()
        title = f"{category_col} distribution (counts)"

    if grouped.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 4))
    if chart_type == "pie":
        ax.pie(grouped.values, labels=grouped.index, autopct="%1.1f%%", startangle=90)
        ax.set_title(title)
    else:
        grouped.head(20).plot(kind="bar", ax=ax, color="#ff7f0e")
        ax.set_ylabel("Value")
        ax.set_title(title)
        ax.set_xticklabels(grouped.head(20).index, rotation=45, ha="right")
    return fig


def scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
) -> Optional[plt.Figure]:
    if not x_col or not y_col:
        return None
    if x_col not in df.columns or y_col not in df.columns:
        return None
    data = df.copy()
    data[x_col] = pd.to_numeric(data[x_col], errors="coerce")
    data[y_col] = pd.to_numeric(data[y_col], errors="coerce")
    data = data.dropna(subset=[x_col, y_col])
    if data.empty:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    if color_col and color_col in df.columns:
        scatter = ax.scatter(
            data[x_col],
            data[y_col],
            c=pd.Categorical(data[color_col]).codes,
            cmap="viridis",
            alpha=0.7,
        )
        ax.legend(*scatter.legend_elements(), title=color_col, loc="best")
    else:
        ax.scatter(data[x_col], data[y_col], alpha=0.7, color="#2ca02c")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} vs {x_col}")
    ax.grid(True, alpha=0.3)
    return fig


def correlation_heatmap(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Optional[plt.Figure]:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return None
    if columns:
        valid_cols = [col for col in columns if col in numeric_df.columns]
        if len(valid_cols) >= 2:
            numeric_df = numeric_df[valid_cols]
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    return fig
