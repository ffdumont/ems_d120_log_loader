# ems_d120_log_loader

A command-line toolkit for processing, analyzing, and visualizing Dynon EMS-D120 (and compatible) flight log data. Supports full workflow from raw CSV import to per-flight statistics, charts, and Obsidian-ready Markdown/HTML summaries and indexes.

---

## Features & Capabilities

- **CSV to SQLite**: Import Dynon EMS-D120 CSV logs into a normalized SQLite database with schema inference and data cleaning.
- **Flight Segmentation**: Automatically segment flights from continuous logs.
- **Batch Chart & Statistics Generation**: Generate charts (PNG), statistics (CSV), and Markdown/HTML summaries for single or multiple flights, with all outputs organized in per-flight subfolders.
- **Configurable Charting**: Unified `config.json` for all chart specs, smoothing, axis grouping, and per-series color.
- **Flexible Flight Selection**: Process a single flight, a range, or all flights in batch mode.
- **Obsidian/Markdown Publishing**: Generate per-flight Markdown summaries and a Markdown index with Obsidian-style links for easy integration into your knowledge base.
- **HTML Reports & Indexes**: Optionally generate HTML summaries and indexes for browser-based review.
- **Output Organization**: All outputs (PNG, CSV, HTML, MD) are organized in per-flight subfolders under the configured output directory.
- **No Console Stats**: All statistics are exported to CSV only (no console output).
- **Cross-platform**: Works on Windows, macOS, Linux (Python 3.8+).

---

## Installation

```bash
# Clone and install dependencies
cd <your_workspace>
pip install -r requirements.txt
```

---

## Configuration

All workflow settings are managed in `config.json`:
- `input_path`: Directory for raw Dynon CSV files
- `output_path`: Directory for all generated outputs (charts, reports, summaries)
- `charts`: Chart definitions, axis grouping, smoothing, and color
- `output_fields`: Fields to export in summary CSVs

Example:
```json
{
  "input_path": "g:/.../input/",
  "output_path": "g:/.../output/",
  ...
}
```

---

## Workflow & Usage

### 1. Import CSV to SQLite

Convert Dynon CSV logs to a SQLite database:
```powershell
python dynon_csv_to_sqlite.py --input <csv_folder> --output <output_db.sqlite>
```
- Input folder and output DB can be set in `config.json` or via CLI.

### 2. Segment Flights

Automatically segment flights:
```powershell
python segment_flights.py --db <output_db.sqlite>
```

### 3. Generate Charts, Stats, and Summaries

Generate all charts, stats, and Markdown/HTML summaries for a single flight, range, or all flights:
```powershell
python generate_chart.py --flight 4 6 --chart all --analyze --markdown --html
```
- All outputs go to the `output_path` directory, organized as `output_path/flight_XXX/`.
- Supported options:
  - `--flight <N>`: Single flight (e.g. 4)
  - `--flight <N1> <N2>`: Range (e.g. 4 6)
  - `--flight all`: All flights
  - `--chart <type>`: Chart type from config, or `all` for all
  - `--analyze`: Export statistics to CSV
  - `--markdown`: Generate Markdown summary for each flight
  - `--html`: Generate HTML summary for each flight

### 4. Generate Indexes

Create Markdown and/or HTML index of all flights:
```powershell
python generate_index.py --markdown --html
```
- Indexes are written to the root of `output_path`.
- Markdown index uses Obsidian-style links: `[[flight_004_summary]]`

---

## Output Structure

```
output_path/
  flight_004/
    flight_004_engine_performance.png
    flight_004_engine_performance.csv
    ...
    flight_004_summary.md
    flight_004_summary.html
  flight_005/
    ...
  index.md
  index.html
```

---

## User Manual & Tips

- **Config-Driven**: All charting, smoothing, and output options are set in `config.json`.
- **Batch Mode**: Use `--chart all` and a flight range or `all` for full automation.
- **Obsidian Integration**: Use the Markdown summaries and index for seamless integration into your Obsidian vault.
- **Custom Charts**: Add or modify chart definitions in `config.json` to suit your analysis needs.
- **No Console Stats**: All statistics are written to CSV files in each flight's folder.
- **Troubleshooting**: If outputs are missing, check `output_path` in `config.json` and ensure you have write permissions.

---

## Requirements
- Python 3.8+
- matplotlib, numpy (see `requirements.txt`)

---

## License
MIT
