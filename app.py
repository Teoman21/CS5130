from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
import pandas as pd

from data_processor import (
    filter_data,
    get_categorical_summary,
    get_column_types,
    get_correlation_matrix,
    get_data_preview,
    get_dataset_overview,
    get_missing_value_report,
    get_numerical_summary,
    load_dataset,
)
from insights import generate_insights
from utils import dataframe_to_csv_bytes, fig_to_png_bytes, format_overview
from visualizations import (
    AGGREGATION_METHODS,
    category_plot,
    correlation_heatmap,
    distribution_plot,
    scatter_plot,
    time_series_plot,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
SAMPLE_FILES = sorted([p.name for p in DATA_DIR.glob("*.csv")])


def _default_dropdown_update(options: List[str]) -> Tuple[gr.Dropdown.update, Optional[str]]:
    value = options[0] if options else None
    return gr.Dropdown.update(choices=options, value=value), value


def load_data(sample_name: Optional[str], uploaded_file, preview_rows: int):
    try:
        df = load_dataset(uploaded_file, sample_name)
    except ValueError as err:
        empty_frame = pd.DataFrame()
        empty_download = gr.DownloadButton.update(visible=False, value=None)
        return (
            None,
            f"❌ {err}",
            "No dataset loaded yet.",
            empty_frame,
            empty_frame,
            empty_frame,
            empty_frame,
            empty_frame,
            empty_frame,
            gr.Dropdown.update(choices=[]),
            gr.Number.update(value=None),
            gr.Number.update(value=None),
            gr.Dropdown.update(choices=[]),
            gr.Dropdown.update(choices=[], value=[]),
            gr.Dropdown.update(choices=[]),
            gr.Date.update(value=None),
            gr.Date.update(value=None),
            empty_frame,
            "Filtered rows: 0",
            empty_download,
            gr.Dropdown.update(choices=[]),
            gr.Dropdown.update(choices=[]),
            gr.Dropdown.update(choices=[]),
            gr.Dropdown.update(choices=[]),
            gr.Dropdown.update(choices=[]),
            gr.Dropdown.update(choices=[]),
            gr.Dropdown.update(choices=[]),
            gr.Dropdown.update(choices=[]),
            gr.Dropdown.update(choices=[]),
            gr.Dropdown.update(choices=[]),
            gr.Dropdown.update(choices=[]),
        )

    overview = get_dataset_overview(df)
    head_df, tail_df = get_data_preview(df, rows=preview_rows)
    numeric_summary = get_numerical_summary(df)
    categorical_summary = get_categorical_summary(df)
    missing_report = get_missing_value_report(df)
    corr_matrix = get_correlation_matrix(df)
    column_types = get_column_types(df)
    numeric_cols = column_types["numeric"]
    categorical_cols = column_types["categorical"]
    date_cols = column_types["date"]
    all_columns = df.columns.tolist()

    numeric_update, numeric_default = _default_dropdown_update(numeric_cols)
    categorical_update, categorical_default = _default_dropdown_update(categorical_cols)
    date_update, date_default = _default_dropdown_update(date_cols)
    filter_preview = df.head(50)
    filter_row_count = f"Filtered rows: {len(df)}"
    download_update = gr.DownloadButton.update(
        value=dataframe_to_csv_bytes(df),
        visible=True,
        label="Download filtered CSV",
    )

    numeric_min_update, numeric_max_update = _numeric_bounds(df, numeric_default)
    cat_values_update = _categorical_values(df, categorical_default)
    date_start_update, date_end_update = _date_bounds(df, date_default)

    status = f"✅ Loaded dataset with {len(df)} rows and {df.shape[1]} columns."

    ts_date_update = _default_dropdown_update(date_cols)[0]
    ts_value_update = _default_dropdown_update(numeric_cols)[0]
    dist_update = _default_dropdown_update(numeric_cols)[0]
    category_col_update = _default_dropdown_update(categorical_cols)[0]
    category_value_update = _default_dropdown_update(numeric_cols)[0]
    scatter_x_update = _default_dropdown_update(numeric_cols)[0]
    scatter_y_update = _default_dropdown_update(numeric_cols)[0]
    scatter_color_update = gr.Dropdown.update(choices=all_columns, value=None)
    heatmap_update = gr.Dropdown.update(
        choices=numeric_cols,
        value=numeric_cols[: min(len(numeric_cols), 3)],
    )
    insights_numeric_update = _default_dropdown_update(numeric_cols)[0]
    insights_date_update = _default_dropdown_update(date_cols)[0]
    insights_value_update = _default_dropdown_update(numeric_cols)[0]

    return (
        df,
        status,
        format_overview(overview),
        head_df,
        tail_df,
        numeric_summary,
        categorical_summary,
        missing_report,
        corr_matrix,
        numeric_update,
        numeric_min_update,
        numeric_max_update,
        categorical_update,
        cat_values_update,
        date_update,
        date_start_update,
        date_end_update,
        filter_preview,
        filter_row_count,
        download_update,
        ts_date_update,
        ts_value_update,
        dist_update,
        category_col_update,
        category_value_update,
        scatter_x_update,
        scatter_y_update,
        scatter_color_update,
        heatmap_update,
        insights_numeric_update,
        insights_date_update,
        insights_value_update,
    )


def _numeric_bounds(df: Optional[pd.DataFrame], column: Optional[str]) -> Tuple[gr.Number.update, gr.Number.update]:
    if df is None or not column or column not in df.columns:
        return gr.Number.update(value=None), gr.Number.update(value=None)
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return gr.Number.update(value=None), gr.Number.update(value=None)
    return (
        gr.Number.update(value=float(series.min())),
        gr.Number.update(value=float(series.max())),
    )


def _categorical_values(df: Optional[pd.DataFrame], column: Optional[str]) -> gr.Dropdown.update:
    if df is None or not column or column not in df.columns:
        return gr.Dropdown.update(choices=[], value=[])
    unique_values = df[column].dropna().astype(str).unique().tolist()
    return gr.Dropdown.update(choices=unique_values, value=unique_values[: min(len(unique_values), 10)])


def _date_bounds(df: Optional[pd.DataFrame], column: Optional[str]) -> Tuple[gr.Date.update, gr.Date.update]:
    if df is None or not column or column not in df.columns:
        return gr.Date.update(value=None), gr.Date.update(value=None)
    series = pd.to_datetime(df[column], errors="coerce").dropna()
    if series.empty:
        return gr.Date.update(value=None), gr.Date.update(value=None)
    return (
        gr.Date.update(value=series.min().date()),
        gr.Date.update(value=series.max().date()),
    )


def update_numeric_bounds(df: Optional[pd.DataFrame], column: Optional[str]):
    return _numeric_bounds(df, column)


def update_categorical_values(df: Optional[pd.DataFrame], column: Optional[str]):
    return _categorical_values(df, column)


def update_date_bounds(df: Optional[pd.DataFrame], column: Optional[str]):
    return _date_bounds(df, column)


def apply_filters(
    df: Optional[pd.DataFrame],
    numeric_col: Optional[str],
    numeric_min: Optional[float],
    numeric_max: Optional[float],
    categorical_col: Optional[str],
    categorical_values: Optional[List[str]],
    date_col: Optional[str],
    start_date,
    end_date,
):
    if df is None:
        empty = pd.DataFrame()
        return empty, "Filtered rows: 0", gr.DownloadButton.update(visible=False)

    numeric_filters = {}
    if numeric_col:
        numeric_filters[numeric_col] = (numeric_min, numeric_max)
    categorical_filters = {}
    if categorical_col and categorical_values:
        categorical_filters[categorical_col] = categorical_values
    date_filters = {}
    if date_col:
        start = start_date.isoformat() if hasattr(start_date, "isoformat") else start_date
        end = end_date.isoformat() if hasattr(end_date, "isoformat") else end_date
        date_filters[date_col] = (start, end)

    filtered = filter_data(
        df,
        numeric_filters=numeric_filters,
        categorical_filters=categorical_filters,
        date_filters=date_filters,
    )
    filtered_preview = filtered.head(50)
    row_count = f"Filtered rows: {len(filtered)}"
    download_update = gr.DownloadButton.update(
        value=dataframe_to_csv_bytes(filtered),
        visible=len(filtered) > 0,
        label="Download filtered CSV",
    )
    return filtered_preview, row_count, download_update


def render_time_series(df: Optional[pd.DataFrame], date_col, value_col, agg_method):
    fig = time_series_plot(df, date_col, value_col, agg_method) if df is not None else None
    png_bytes = fig_to_png_bytes(fig)
    download_update = gr.DownloadButton.update(
        value=png_bytes,
        visible=png_bytes is not None,
        label="Download time-series PNG",
    )
    return fig, download_update


def render_distribution(df: Optional[pd.DataFrame], column, plot_type):
    return distribution_plot(df, column, plot_type) if df is not None else None


def render_category(df: Optional[pd.DataFrame], cat_col, value_col, agg_method, chart_type):
    return category_plot(df, cat_col, value_col, agg_method, chart_type) if df is not None else None


def render_scatter(df: Optional[pd.DataFrame], x_col, y_col, color_col):
    return scatter_plot(df, x_col, y_col, color_col) if df is not None else None


def render_heatmap(df: Optional[pd.DataFrame], columns: Optional[List[str]]):
    return correlation_heatmap(df, columns) if df is not None else None


def render_insights(df: Optional[pd.DataFrame], numeric_col, date_col, value_col):
    if df is None:
        empty = pd.DataFrame()
        return empty, empty, "Load a dataset to see insights."
    result = generate_insights(df, numeric_col, date_col, value_col)
    top_df = result["top"]
    bottom_df = result["bottom"]
    trend_summary = result["trend"]
    return top_df, bottom_df, trend_summary


with gr.Blocks(title="Walmart Sales Explorer") as demo:
    gr.Markdown(
        """
        # Walmart Sales Forecast Explorer
        Upload your own CSV/Excel file or start with one of the sample Walmart datasets. 
        Explore statistics, filter dynamically, visualize trends, and export insights.
        """
    )

    data_state = gr.State()

    with gr.Row():
        sample_dropdown = gr.Dropdown(
            label="Sample datasets",
            choices=SAMPLE_FILES,
            value=SAMPLE_FILES[0] if SAMPLE_FILES else None,
            interactive=True,
        )
        file_input = gr.File(label="Upload CSV or Excel", file_types=[".csv", ".xls", ".xlsx"])
        preview_rows = gr.Slider(
            label="Preview rows",
            minimum=5,
            maximum=50,
            step=5,
            value=10,
        )
        load_button = gr.Button("Load data", variant="primary")

    status_md = gr.Markdown("No dataset loaded yet.")

    with gr.Tabs():
        with gr.Tab("Data Overview"):
            overview_md = gr.Markdown()
            with gr.Row():
                head_table = gr.Dataframe(label="Head")
                tail_table = gr.Dataframe(label="Tail")
            gr.Markdown("### Summary Statistics")
            with gr.Row():
                numeric_stats = gr.Dataframe(label="Numerical summary")
                categorical_stats = gr.Dataframe(label="Categorical summary")
            with gr.Row():
                missing_report_table = gr.Dataframe(label="Missing values")
                corr_table = gr.Dataframe(label="Correlation matrix")

        with gr.Tab("Filtering"):
            gr.Markdown("Use the controls below to filter rows. The preview updates immediately.")
            with gr.Row():
                numeric_filter_col = gr.Dropdown(label="Numeric column", choices=[])
                numeric_min = gr.Number(label="Minimum")
                numeric_max = gr.Number(label="Maximum")
            with gr.Row():
                categorical_filter_col = gr.Dropdown(label="Categorical column", choices=[])
                categorical_filter_values = gr.Dropdown(
                    label="Categories", choices=[], multiselect=True
                )
            with gr.Row():
                date_filter_col = gr.Dropdown(label="Date column", choices=[])
                date_start = gr.Date(label="Start date")
                date_end = gr.Date(label="End date")
            apply_filters_btn = gr.Button("Apply filters", variant="secondary")
            filtered_count_md = gr.Markdown("Filtered rows: 0")
            filtered_preview = gr.Dataframe(label="Filtered preview")
            filtered_download_btn = gr.DownloadButton(
                label="Download filtered CSV",
                visible=False,
                file_name="filtered_data.csv",
            )

        with gr.Tab("Visualizations"):
            gr.Markdown("Create interactive plots by selecting the columns you want to explore.")
            gr.Markdown("#### Time Series")
            with gr.Row():
                ts_date_col = gr.Dropdown(label="Date column", choices=[])
                ts_value_col = gr.Dropdown(label="Value column", choices=[])
                ts_agg_method = gr.Dropdown(
                    label="Aggregation",
                    choices=list(AGGREGATION_METHODS.keys()),
                    value="sum",
                )
            time_series_plot_comp = gr.Matplotlib(label="Time series")
            time_series_download_btn = gr.DownloadButton(
                label="Download time-series PNG",
                visible=False,
                file_name="time_series.png",
            )

            gr.Markdown("#### Distribution")
            with gr.Row():
                dist_column = gr.Dropdown(label="Numeric column", choices=[])
                dist_type = gr.Radio(
                    label="Plot type",
                    choices=["histogram", "box"],
                    value="histogram",
                )
            distribution_plot_comp = gr.Matplotlib(label="Distribution plot")

            gr.Markdown("#### Category Analysis")
            with gr.Row():
                category_column = gr.Dropdown(label="Category column", choices=[])
                category_value_column = gr.Dropdown(label="Value column", choices=[])
                category_agg_method = gr.Dropdown(
                    label="Aggregation",
                    choices=list(AGGREGATION_METHODS.keys()),
                    value="sum",
                )
                category_chart_type = gr.Radio(
                    label="Chart type",
                    choices=["bar", "pie"],
                    value="bar",
                )
            category_plot_comp = gr.Matplotlib(label="Category plot")

            gr.Markdown("#### Scatter Plot")
            with gr.Row():
                scatter_x = gr.Dropdown(label="X axis", choices=[])
                scatter_y = gr.Dropdown(label="Y axis", choices=[])
                scatter_color = gr.Dropdown(label="Color (optional)", choices=[])
            scatter_plot_comp = gr.Matplotlib(label="Scatter plot")

            gr.Markdown("#### Correlation Heatmap")
            heatmap_columns = gr.Dropdown(
                label="Columns", choices=[], multiselect=True
            )
            heatmap_plot_comp = gr.Matplotlib(label="Correlation heatmap")

        with gr.Tab("Insights & Export"):
            gr.Markdown("Automatically surface top/bottom performers and trend insights.")
            insights_numeric_col = gr.Dropdown(label="Numeric column", choices=[])
            insights_date_col = gr.Dropdown(label="Date column", choices=[])
            insights_value_col = gr.Dropdown(label="Value column", choices=[])
            with gr.Row():
                top_table = gr.Dataframe(label="Top performers")
                bottom_table = gr.Dataframe(label="Bottom performers")
            insights_trend_md = gr.Markdown("Load data to generate insights.")

    load_button.click(
        fn=load_data,
        inputs=[sample_dropdown, file_input, preview_rows],
        outputs=[
            data_state,
            status_md,
            overview_md,
            head_table,
            tail_table,
            numeric_stats,
            categorical_stats,
            missing_report_table,
            corr_table,
            numeric_filter_col,
            numeric_min,
            numeric_max,
            categorical_filter_col,
            categorical_filter_values,
            date_filter_col,
            date_start,
            date_end,
            filtered_preview,
            filtered_count_md,
            filtered_download_btn,
            ts_date_col,
            ts_value_col,
            dist_column,
            category_column,
            category_value_column,
            scatter_x,
            scatter_y,
            scatter_color,
            heatmap_columns,
            insights_numeric_col,
            insights_date_col,
            insights_value_col,
        ],
    )

    numeric_filter_col.change(
        fn=update_numeric_bounds,
        inputs=[data_state, numeric_filter_col],
        outputs=[numeric_min, numeric_max],
    )

    categorical_filter_col.change(
        fn=update_categorical_values,
        inputs=[data_state, categorical_filter_col],
        outputs=[categorical_filter_values],
    )

    date_filter_col.change(
        fn=update_date_bounds,
        inputs=[data_state, date_filter_col],
        outputs=[date_start, date_end],
    )

    apply_filters_btn.click(
        fn=apply_filters,
        inputs=[
            data_state,
            numeric_filter_col,
            numeric_min,
            numeric_max,
            categorical_filter_col,
            categorical_filter_values,
            date_filter_col,
            date_start,
            date_end,
        ],
        outputs=[filtered_preview, filtered_count_md, filtered_download_btn],
    )

    ts_inputs = [data_state, ts_date_col, ts_value_col, ts_agg_method]
    load_button.click(fn=render_time_series, inputs=ts_inputs, outputs=[time_series_plot_comp, time_series_download_btn])
    ts_date_col.change(fn=render_time_series, inputs=ts_inputs, outputs=[time_series_plot_comp, time_series_download_btn])
    ts_value_col.change(fn=render_time_series, inputs=ts_inputs, outputs=[time_series_plot_comp, time_series_download_btn])
    ts_agg_method.change(fn=render_time_series, inputs=ts_inputs, outputs=[time_series_plot_comp, time_series_download_btn])

    dist_inputs = [data_state, dist_column, dist_type]
    load_button.click(fn=render_distribution, inputs=dist_inputs, outputs=distribution_plot_comp)
    dist_column.change(fn=render_distribution, inputs=dist_inputs, outputs=distribution_plot_comp)
    dist_type.change(fn=render_distribution, inputs=dist_inputs, outputs=distribution_plot_comp)

    cat_inputs = [data_state, category_column, category_value_column, category_agg_method, category_chart_type]
    load_button.click(fn=render_category, inputs=cat_inputs, outputs=category_plot_comp)
    category_column.change(fn=render_category, inputs=cat_inputs, outputs=category_plot_comp)
    category_value_column.change(fn=render_category, inputs=cat_inputs, outputs=category_plot_comp)
    category_agg_method.change(fn=render_category, inputs=cat_inputs, outputs=category_plot_comp)
    category_chart_type.change(fn=render_category, inputs=cat_inputs, outputs=category_plot_comp)

    scatter_inputs = [data_state, scatter_x, scatter_y, scatter_color]
    load_button.click(fn=render_scatter, inputs=scatter_inputs, outputs=scatter_plot_comp)
    scatter_x.change(fn=render_scatter, inputs=scatter_inputs, outputs=scatter_plot_comp)
    scatter_y.change(fn=render_scatter, inputs=scatter_inputs, outputs=scatter_plot_comp)
    scatter_color.change(fn=render_scatter, inputs=scatter_inputs, outputs=scatter_plot_comp)

    heatmap_inputs = [data_state, heatmap_columns]
    load_button.click(fn=render_heatmap, inputs=heatmap_inputs, outputs=heatmap_plot_comp)
    heatmap_columns.change(fn=render_heatmap, inputs=heatmap_inputs, outputs=heatmap_plot_comp)

    insights_inputs = [data_state, insights_numeric_col, insights_date_col, insights_value_col]
    load_button.click(fn=render_insights, inputs=insights_inputs, outputs=[top_table, bottom_table, insights_trend_md])
    for comp in (insights_numeric_col, insights_date_col, insights_value_col):
        comp.change(fn=render_insights, inputs=insights_inputs, outputs=[top_table, bottom_table, insights_trend_md])

if __name__ == "__main__":
    demo.launch()
