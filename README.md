## Walmart Sales Forecast Explorer

This Gradio application lets you explore the Walmart Sales Forecasting dataset (and any CSV/Excel file) through automated profiling, flexible filtering, interactive charts, and basic insight generation. The template follows the provided project structure so you can extend or customize it easily.

### Features

- **Data upload & validation** – upload CSV/XLS/XLSX files or pick from two bundled samples in `data/`.
- **Automated profiling** – numerical stats, categorical summaries, missing-value report, and correlation matrix.
- **Interactive filters** – numeric min/max inputs, categorical multi-selects, and date pickers with real-time row counts.
- **Visualizations** – time series (with aggregation selector), histogram/box plot, category bar/pie chart, scatter plot, and correlation heatmap.
- **Insights** – automatic top/bottom performers plus a basic trend/anomaly detector.
- **Exports** – download filtered rows as CSV and save the time-series plot as a PNG.

### Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the app**
   ```bash
   python app.py
   ```
3. Open the local Gradio link in your browser.

### Project Structure

```
├── app.py                  # Main Gradio application
├── data_processor.py       # Data loading, profiling, filtering utilities
├── insights.py             # Insight generation helpers
├── utils.py                # Shared helpers (formatting, exports)
├── visualizations.py       # Matplotlib chart builders
├── requirements.txt        # Python dependencies
├── README.md               # This guide
└── data/
    ├── sample1.csv         # Walmart train data
    └── sample2.csv         # Walmart features data
```

### Notes

- Replace or extend the sample CSV files in `data/` as needed.
- If you load Excel files, make sure `openpyxl` is installed (already included in `requirements.txt`).
- The app caps previews to keep the interface responsive. Use the CSV export to capture full filtered results.
