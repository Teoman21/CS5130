import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
ALLOWED_EXTENSIONS = {".csv", ".xls", ".xlsx"}


def _resolve_path(sample_name: Optional[str]) -> Optional[Path]:
    if not sample_name:
        return None
    target = DATA_DIR / sample_name
    if not target.exists():
        raise ValueError(f"Sample dataset '{sample_name}' was not found in the data directory.")
    return target


def _load_from_path(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{suffix}'. Please upload CSV or Excel files.")
    if suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def load_dataset(
    uploaded_file: Optional[io.BytesIO],
    sample_name: Optional[str],
) -> pd.DataFrame:
    """
    Load a dataset from a Gradio File input or from the /data directory.
    """
    if uploaded_file is None and not sample_name:
        raise ValueError("Please upload a dataset or choose one of the bundled samples.")

    if uploaded_file is not None:
        path = Path(uploaded_file.name)
        suffix = path.suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise ValueError("Only CSV, XLS, or XLSX files are supported.")
        if suffix == ".csv":
            df = pd.read_csv(uploaded_file.name)
        else:
            df = pd.read_excel(uploaded_file.name)
        return df

    target_path = _resolve_path(sample_name)
    return _load_from_path(target_path)


def get_dataset_overview(df: pd.DataFrame) -> Dict[str, object]:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }


def get_data_preview(df: pd.DataFrame, rows: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = max(1, rows)
    return df.head(rows), df.tail(rows)


def get_numerical_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()
    summary = numeric_df.describe(percentiles=[0.25, 0.5, 0.75]).T
    summary.rename(
        columns={
            "25%": "25%",
            "50%": "median",
            "75%": "75%",
        },
        inplace=True,
    )
    summary["missing"] = df[numeric_df.columns].isna().sum()
    return summary[
        ["count", "mean", "std", "min", "25%", "median", "75%", "max", "missing"]
    ].round(3)


def get_categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    cat_df = df.select_dtypes(include=["object", "category"])
    if cat_df.empty:
        return pd.DataFrame()
    records = []
    for col in cat_df:
        series = cat_df[col].astype(str)
        value_counts = series.value_counts(dropna=True)
        record = {
            "column": col,
            "unique": int(series.nunique()),
            "mode": value_counts.idxmax() if not value_counts.empty else None,
            "mode_count": int(value_counts.iloc[0]) if not value_counts.empty else 0,
            "missing": int(df[col].isna().sum()),
            "sample_values": ", ".join(value_counts.head(5).index.tolist()),
        }
        records.append(record)
    return pd.DataFrame(records)


def get_missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum()
    pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"missing": missing, "percent": pct})
    report = report[report["missing"] > 0]
    return report


def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return pd.DataFrame()
    return numeric_df.corr().round(3)


def get_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
        elif df[col].dtype == "object":
            try:
                pd.to_datetime(df[col], errors="raise")
                date_cols.append(col)
            except (ValueError, TypeError):
                continue
    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "date": date_cols,
    }


def filter_data(
    df: pd.DataFrame,
    numeric_filters: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
    categorical_filters: Optional[Dict[str, List[str]]] = None,
    date_filters: Optional[Dict[str, Tuple[Optional[str], Optional[str]]]] = None,
) -> pd.DataFrame:
    filtered = df.copy()

    numeric_filters = numeric_filters or {}
    for col, bounds in numeric_filters.items():
        if col not in filtered.columns or bounds is None:
            continue
        min_val, max_val = bounds
        if min_val is not None:
            filtered = filtered[filtered[col] >= min_val]
        if max_val is not None:
            filtered = filtered[filtered[col] <= max_val]

    categorical_filters = categorical_filters or {}
    for col, values in categorical_filters.items():
        if col not in filtered.columns or not values:
            continue
        filtered = filtered[filtered[col].isin(values)]

    date_filters = date_filters or {}
    for col, bounds in date_filters.items():
        if col not in filtered.columns or bounds is None:
            continue
        start, end = bounds
        series = pd.to_datetime(filtered[col], errors="coerce")
        if start:
            filtered = filtered[series >= pd.to_datetime(start)]
        if end:
            filtered = filtered[series <= pd.to_datetime(end)]

    return filtered
