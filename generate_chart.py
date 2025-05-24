import argparse
import json
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

DB_PATH = 'logs_test.sqlite'
SEGMENTS_TABLE = 'flight_segments'
LOG_TABLE = 'flight_log'
CHART_SPEC_PATH = 'chart_spec.json'


def parse_args():
    parser = argparse.ArgumentParser(description="Generate flight chart from SQLite data and chart spec.")
    parser.add_argument('--flight', type=int, required=True, help='Flight ID (from flight_segments)')
    parser.add_argument('--chart', type=str, required=True, help='Chart type (from chart_spec.json)')
    parser.add_argument('--output', type=str, help='Output PNG file (optional)')
    parser.add_argument('--analyze', action='store_true', help='Print statistical summary for each Y variable')
    parser.add_argument('--csv', type=str, help='Export statistical summary to CSV file (optional)')
    parser.add_argument('--smooth', action='store_true', help='Apply centered moving average smoothing to plot data')
    parser.add_argument('--window-size', type=int, default=5, help='Window size for smoothing (default: 5)')
    return parser.parse_args()


def load_chart_spec(chart_type):
    with open(CHART_SPEC_PATH, encoding='utf-8') as f:
        spec = json.load(f)
    charts = spec.get('charts', {})
    if chart_type not in charts:
        raise ValueError(f"Chart type '{chart_type}' not found in chart_spec.json")
    return charts[chart_type]


def get_flight_segment(conn, flight_id):
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {SEGMENTS_TABLE} WHERE flight_id=?", (flight_id,))
    row = cur.fetchone()
    if not row:
        raise ValueError(f"Flight ID {flight_id} not found in {SEGMENTS_TABLE}")
    columns = [desc[0] for desc in cur.description]
    return dict(zip(columns, row))


def get_flight_data(conn, start_rowid, end_rowid, y_columns):
    cur = conn.cursor()
    # Always fetch zulu_year, zulu_mo, zulu_day, zulu_hour, zulu_min, zulu_sec for time axis
    cols = [*y_columns, 'zulu_year', 'zulu_mo', 'zulu_day', 'zulu_hour', 'zulu_min', 'zulu_sec']
    quoted_cols = ', '.join([f'"{c}"' for c in cols])
    sql = f"SELECT rowid, {quoted_cols} FROM {LOG_TABLE} WHERE rowid >= ? AND rowid <= ? ORDER BY rowid"
    cur.execute(sql, (start_rowid, end_rowid))
    return cur.fetchall(), [d[0] for d in cur.description]


def reconstruct_time(row, col_idx):
    try:
        y = int(row[col_idx['zulu_year']])
        if y < 100:
            y += 2000
        t = datetime(
            y,
            int(row[col_idx['zulu_mo']]),
            int(row[col_idx['zulu_day']]),
            int(row[col_idx['zulu_hour']]),
            int(row[col_idx['zulu_min']]),
            int(row[col_idx['zulu_sec']])
        )
        return t
    except Exception:
        return None


def analyze_peaks(data: list[float]) -> dict:
    arr = np.array([v for v in data if v is not None])
    if arr.size == 0:
        return {}
    stats = {}
    stats['count'] = int(arr.size)
    stats['mean'] = float(np.mean(arr))
    stats['median'] = float(np.median(arr))
    stats['min'] = float(np.min(arr))
    stats['max'] = float(np.max(arr))
    stats['std'] = float(np.std(arr))
    stats['q1'] = float(np.percentile(arr, 25))
    stats['q3'] = float(np.percentile(arr, 75))
    stats['iqr'] = stats['q3'] - stats['q1']
    q3p = stats['q3'] + 1.5 * stats['iqr']
    q1m = stats['q1'] - 1.5 * stats['iqr']
    stats['spikes'] = int(np.sum(arr > q3p))
    stats['dips'] = int(np.sum(arr < q1m))
    stats['gt_mean_2std'] = int(np.sum(arr > stats['mean'] + 2*stats['std']))
    stats['lt_mean_2std'] = int(np.sum(arr < stats['mean'] - 2*stats['std']))
    stats['zero'] = int(np.sum(arr == 0))
    stats['outliers'] = int(np.sum((arr < q1m) | (arr > q3p)))
    return stats


def analyze_chart_data(data_dict: dict) -> dict:
    return {k: analyze_peaks(v) for k, v in data_dict.items()}


def print_stats_table(stats: dict, csv_path: str = None, flight_info: dict = None):
    headers = [
        'Parameter', 'Count', 'Mean', 'Median', 'Min', 'Max', 'Std', 'Q1', 'Q3', 'IQR',
        'Zero', '>Mean+2Std', '<Mean-2Std', 'Outliers'
    ]
    fmt = "{:<18} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>6} {:>10} {:>10} {:>9}"
    rows = []
    for k, v in stats.items():
        if not v:
            continue
        def fmt_num(val):
            try:
                # Format float with 3 decimals, use comma as decimal separator
                return f"{float(val):.3f}".replace('.', ',')
            except Exception:
                return val
        row = [
            k,
            v.get('count', ''),
            fmt_num(v.get('mean', 0)),
            fmt_num(v.get('median', 0)),
            fmt_num(v.get('min', 0)),
            fmt_num(v.get('max', 0)),
            fmt_num(v.get('std', 0)),
            fmt_num(v.get('q1', 0)),
            fmt_num(v.get('q3', 0)),
            fmt_num(v.get('iqr', 0)),
            v.get('zero', 0),
            v.get('gt_mean_2std', 0),
            v.get('lt_mean_2std', 0),
            v.get('outliers', 0),
        ]
        rows.append(row)
    # Print table
    print(fmt.format(*headers))
    for row in rows:
        print(fmt.format(*row))
    # Optionally export to CSV
    if csv_path:
        import csv
        headers_csv = [h.replace('σ', 'std').replace('Σ', 'Sum') for h in headers]
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f, delimiter=';')
            # Write flight info as first row if provided
            if flight_info:
                flight_row = [
                    f"flight_id: {flight_info.get('flight_id', '')}",
                    f"start_timestamp: {flight_info.get('start_timestamp', '')}",
                    f"end_timestamp: {flight_info.get('end_timestamp', '')}"
                ]
                writer.writerow(flight_row)
            writer.writerow(headers_csv)
            for row in rows:
                row_csv = [str(cell).replace('.', ',') if isinstance(cell, float) or (isinstance(cell, str) and cell.replace(',', '').replace('.', '').isdigit() and '.' in cell) else str(cell) for cell in row]
                writer.writerow(row_csv)
        print(f"Statistical summary exported to {csv_path}")


def moving_average_centered(data, window_size=5):
    """Apply centered moving average smoothing, skipping None values."""
    if window_size < 2:
        return data[:]
    n = len(data)
    half = window_size // 2
    smoothed = []
    for i in range(n):
        # Determine window bounds
        start = max(0, i - half)
        end = min(n, i + half + 1)
        window = [x for x in data[start:end] if x is not None]
        if window:
            smoothed.append(sum(window) / len(window))
        else:
            smoothed.append(data[i])  # fallback to original if all None
    return smoothed


def main():
    args = parse_args()
    chart_spec = load_chart_spec(args.chart)
    y_axes = chart_spec['y_axes']
    y_columns = [y['column'] for y in y_axes]
    y_labels = [y['label'] for y in y_axes]
    conn = sqlite3.connect(DB_PATH)
    flight = get_flight_segment(conn, args.flight)
    start_rowid = flight['start_rowid']
    end_rowid = flight['end_rowid']
    data, colnames = get_flight_data(conn, start_rowid, end_rowid, y_columns)
    col_idx = {name: i+1 for i, name in enumerate(y_columns)}  # +1 because rowid is at 0
    # Add time columns
    for tcol in ['zulu_year', 'zulu_mo', 'zulu_day', 'zulu_hour', 'zulu_min', 'zulu_sec']:
        col_idx[tcol] = colnames.index(tcol)
    # Build time axis
    times = []
    y_data = [[] for _ in y_columns]
    t0 = None
    for row in data:
        t = reconstruct_time(row, col_idx)
        if t is None:
            continue
        if t0 is None:
            t0 = t
        dt_min = (t - t0).total_seconds() / 60.0
        times.append(dt_min)
        for i, ycol in enumerate(y_columns):
            val = row[col_idx[ycol]]
            try:
                y_data[i].append(float(str(val).replace(',', '.')) if val not in (None, '', 'N/A') else None)
            except Exception:
                y_data[i].append(None)
    # Smoothing (for plot only)
    if hasattr(args, 'smooth') and args.smooth:
        window_size = getattr(args, 'window_size', 5) or 5
        y_data_smoothed = [moving_average_centered(yd, window_size) for yd in y_data]
        plot_y_data = y_data_smoothed
        title_suffix = " (smoothed)"
    else:
        plot_y_data = y_data
        title_suffix = ""
    # Plot with support for multiple y-axes
    # Prepare axis assignment and colors
    axis_indices = [y.get('axis', i) for i, y in enumerate(y_axes)]
    axis_colors = ['tab:blue', 'tab:orange', 'tab:green']
    axes = [None, None, None]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    axes[0] = ax1  # Primary left Y-axis
    ax2 = ax1.twinx()
    axes[1] = ax2
    ax3 = None
    if 2 in axis_indices:
        ax3 = ax1.twinx()
        axes[2] = ax3
        ax3.spines['right'].set_position(('axes', 1.15))  # Second right Y-axis, offset further right
    ax2.spines['right'].set_position(('axes', 1.0))  # First right Y-axis
    # Plot each series on its assigned axis
    line_handles = []
    for i, y in enumerate(plot_y_data):
        axis_idx = axis_indices[i]
        ax = axes[axis_idx]
        color = axis_colors[axis_idx % len(axis_colors)]
        l, = ax.plot(times, y, label=y_labels[i], color=color)
        line_handles.append(l)
        # Set y-label color and tick color
        ax.yaxis.label.set_color(color)
        ax.tick_params(axis='y', colors=color)
    # Set axis labels
    ax1.set_xlabel('Time since start of flight (min)')
    if axes[0]:
        axes[0].set_ylabel(y_labels[axis_indices.index(0)])
    if axes[1]:
        axes[1].set_ylabel(y_labels[axis_indices.index(1)])
    if axes[2]:
        axes[2].set_ylabel(y_labels[axis_indices.index(2)])
    # Title
    title = f"Flight {args.flight}: {chart_spec.get('label', args.chart)}{title_suffix}\n{flight['start_timestamp']} to {flight['end_timestamp']}"
    plt.title(title)
    # Combine all legend entries
    labs = [l.get_label() for l in line_handles]
    plt.legend(line_handles, labs, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    # Save chart as PNG automatically if --output not specified
    output_path = args.output
    if not output_path:
        output_path = f"chart_{args.chart}_flight{args.flight}.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Chart saved to {output_path}")
    if not args.output:
        plt.show()
    # Optional analysis
    if hasattr(args, 'analyze') and args.analyze:
        data_dict = {ycol: y_data[i] for i, ycol in enumerate(y_columns)}
        stats = analyze_chart_data(data_dict)
        print("\nStatistical summary:")
        # Determine CSV path
        csv_path = None
        if hasattr(args, 'csv') and args.csv:
            if args.csv.lower() == 'auto':
                csv_path = f"stats_{args.chart}_flight{args.flight}.csv"
            else:
                csv_path = args.csv
        elif hasattr(args, 'csv'):
            csv_path = f"stats_{args.chart}_flight{args.flight}.csv"
        print_stats_table(stats, csv_path=csv_path, flight_info=flight)
    conn.close()

if __name__ == "__main__":
    # Add --csv, --smooth, --window-size arguments to parser
    import sys
    import argparse as _argparse
    def _patched_parse_args():
        parser = _argparse.ArgumentParser(description="Generate flight chart from SQLite data and chart spec.")
        parser.add_argument('--flight', type=int, required=True, help='Flight ID (from flight_segments)')
        parser.add_argument('--chart', type=str, required=True, help='Chart type (from chart_spec.json)')
        parser.add_argument('--output', type=str, help='Output PNG file (optional)')
        parser.add_argument('--analyze', action='store_true', help='Print statistical summary for each Y variable')
        parser.add_argument('--csv', type=str, help='Export statistical summary to CSV file (optional)')
        parser.add_argument('--smooth', action='store_true', help='Apply centered moving average smoothing to plot data')
        parser.add_argument('--window-size', type=int, default=5, help='Window size for smoothing (default: 5)')
        return parser.parse_args()
    parse_args = _patched_parse_args
    main()
