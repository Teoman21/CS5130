from typing import Dict, Optional

import pandas as pd


def top_bottom_insights(df: pd.DataFrame, column: Optional[str], n: int = 5) -> Dict[str, pd.DataFrame]:
    if not column or column not in df.columns:
        return {"top": pd.DataFrame(), "bottom": pd.DataFrame()}
    numeric_series = pd.to_numeric(df[column], errors="coerce")
    valid = df[numeric_series.notna()].copy()
    if valid.empty:
        return {"top": pd.DataFrame(), "bottom": pd.DataFrame()}
    valid[column] = numeric_series.loc[valid.index]
    top = valid.sort_values(by=column, ascending=False).head(n)
    bottom = valid.sort_values(by=column, ascending=True).head(n)
    return {"top": top, "bottom": bottom}


def detect_trends_or_anomalies(
    df: pd.DataFrame,
    date_col: Optional[str],
    value_col: Optional[str],
) -> str:
    if (
        not date_col
        or not value_col
        or date_col not in df.columns
        or value_col not in df.columns
    ):
        return "Select a date column and a numeric column to analyze trends."

    date_series = pd.to_datetime(df[date_col], errors="coerce")
    numeric_series = pd.to_numeric(df[value_col], errors="coerce")
    valid = df[date_series.notna() & numeric_series.notna()].copy()
    if valid.empty:
        return "Not enough valid rows to detect trends."

    valid[date_col] = date_series.loc[valid.index]
    valid[value_col] = numeric_series.loc[valid.index]
    aggregated = valid.groupby(date_col)[value_col].mean().sort_index()
    if aggregated.shape[0] < 3:
        return "Need at least 3 time points to detect a trend."

    first, last = aggregated.iloc[0], aggregated.iloc[-1]
    pct_change = ((last - first) / first * 100) if first != 0 else None
    trend = "increasing" if last > first else "decreasing"

    anomalies = aggregated[(aggregated - aggregated.mean()).abs() > 2 * aggregated.std()]
    anomaly_message = (
        f"Detected {len(anomalies)} potential anomalies at {list(anomalies.index)}."
        if not anomalies.empty
        else "No major anomalies detected."
    )

    if pct_change is None:
        return f"The series appears to be {trend}. {anomaly_message}"

    return (
        f"The series is {trend} with an overall change of {pct_change:.2f}% from start to finish. "
        + anomaly_message
    )


def generate_insights(
    df: pd.DataFrame,
    numeric_column: Optional[str],
    date_column: Optional[str],
    value_column: Optional[str],
) -> Dict[str, object]:
    extremes = top_bottom_insights(df, numeric_column)
    trend_summary = detect_trends_or_anomalies(df, date_column, value_column)
    return {
        "top": extremes["top"],
        "bottom": extremes["bottom"],
        "trend": trend_summary,
    }
