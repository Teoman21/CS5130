import io
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd


def format_overview(overview: Dict[str, object]) -> str:
    if not overview:
        return "No dataset loaded yet."
    columns = ", ".join(overview.get("column_names", []))
    dtype_lines = "\n".join(
        f"- {col}: {dtype}" for col, dtype in overview.get("dtypes", {}).items()
    )
    return (
        f"**Rows**: {overview.get('rows', 0)}\n"
        f"**Columns**: {overview.get('columns', 0)}\n"
        f"**Column Names**: {columns}\n"
        f"**Data Types**:\n{dtype_lines}"
    )


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 5) -> str:
    if df is None or df.empty:
        return "No data available."
    return df.head(max_rows).to_markdown()


def fig_to_png_bytes(fig: Optional[plt.Figure]) -> Optional[bytes]:
    if fig is None:
        return None
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    return buffer.read()


def dataframe_to_csv_bytes(df: Optional[pd.DataFrame]) -> Optional[bytes]:
    if df is None or df.empty:
        return None
    return df.to_csv(index=False).encode("utf-8")
