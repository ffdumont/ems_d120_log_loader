import argparse
import json
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import sys
import subprocess

DB_PATH = 'logs_test.sqlite'
SEGMENTS_TABLE = 'flight_segments'
LOG_TABLE = 'flight_log'
CONFIG_PATH = 'config.json'


def parse_args():
    parser = argparse.ArgumentParser(description="Generate flight chart from SQLite data and chart spec.")
    parser.add_argument('--flight', nargs='+', required=True, help='Flight ID (single: 3), range: 3 6, or all')
    parser.add_argument('--chart', type=str, required=True, help='Chart type (from config.json) or "all" for all charts')
    parser.add_argument('--output', type=str, help='Output PNG file or output folder (optional)')
    parser.add_argument('--analyze', action='store_true', help='Print statistical summary for each Y variable')
    parser.add_argument('--csv', type=str, help='Export statistical summary to CSV file (optional)')
    return parser.parse_args()


def load_chart_spec(chart_type):
    with open(CONFIG_PATH, encoding='utf-8') as f:
        config = json.load(f)
    charts = config.get('charts', {})
    if chart_type not in charts:
        raise ValueError(f"Chart type '{chart_type}' not found in config.json")
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
    # Only export to CSV, do not print table
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
        # No print statement for CSV export


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
    # Determine flight IDs to process
    if len(args.flight) == 1 and args.flight[0].lower() == 'all':
        # All flights in DB
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(f"SELECT flight_id FROM {SEGMENTS_TABLE} ORDER BY flight_id")
        flight_ids = [row[0] for row in cur.fetchall()]
        conn.close()
    elif len(args.flight) == 2:
        # Range of flights
        start, end = int(args.flight[0]), int(args.flight[1])
        flight_ids = list(range(start, end + 1))
    elif len(args.flight) == 1:
        # Single flight
        flight_ids = [int(args.flight[0])]
    else:
        raise ValueError("Invalid --flight argument. Use a single ID, a range, or 'all'.")

    for flight_id in flight_ids:
        # Format flight folder name as flight_XXX
        flight_folder = f"flight_{int(flight_id):03d}"
        # Determine output base directory
        output_base = args.output or "charts_output"
        if output_base.endswith('.png') or output_base.endswith('.csv'):
            output_base = os.path.dirname(output_base) or "charts_output"
        # Only create one subfolder: charts_output/flight_XXX
        flight_dir = os.path.join(output_base, flight_folder)
        os.makedirs(flight_dir, exist_ok=True)

        if args.chart == "all":
            # Batch mode: generate all charts for this flight
            with open(CONFIG_PATH, encoding='utf-8') as f:
                config = json.load(f)
            charts = config.get('charts', {})
            chart_files = []
            for chart_type, chart_spec in charts.items():
                print(f"Generating chart: {chart_spec.get('label', chart_type)} -> {flight_dir}/{flight_folder}_{chart_type}.png")
                cmd = [
                    sys.executable, __file__,
                    "--flight", str(flight_id),
                    "--chart", chart_type,
                    "--analyze"
                ]
                subprocess.run(cmd, check=True)
                # Collect generated files for HTML
                chart_png = f"{flight_folder}_{chart_type}.png"
                chart_csv = f"{flight_folder}_{chart_type}.csv"
                chart_files.append((chart_type, chart_spec.get('label', chart_type), chart_png, chart_csv))
            # After all charts, generate HTML summary for this flight
            html_path = os.path.join(flight_dir, f"{flight_folder}_summary.html")
            with open(html_path, 'w', encoding='utf-8') as html:
                html.write(f"<html><head><title>Flight {flight_id} Charts & Stats</title><style>body{{font-family:sans-serif;}}table{{border-collapse:collapse;}}th,td{{border:1px solid #ccc;padding:4px;}}img{{max-width:800px;display:block;margin-bottom:8px;}}</style></head><body>")
                html.write(f"<h1>Flight {flight_id} Charts & Statistics</h1>")
                for chart_type, chart_label, chart_png, chart_csv in chart_files:
                    html.write(f"<h2>{chart_label}</h2>")
                    html.write(f'<img src="{chart_png}" alt="{chart_label}">')
                    # Insert stats table from CSV
                    csv_path = os.path.join(flight_dir, chart_csv)
                    if os.path.exists(csv_path):
                        with open(csv_path, encoding='utf-8-sig') as f:
                            lines = f.readlines()
                        # Skip flight info row if present
                        start = 0
                        if lines and lines[0].startswith('flight_id:'):
                            start = 1
                        if len(lines) > start:
                            html.write('<table>')
                            headers = lines[start].strip().split(';')
                            html.write('<tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr>')
                            for line in lines[start+1:]:
                                cells = line.strip().split(';')
                                html.write('<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>')
                            html.write('</table>')
                html.write("</body></html>")
            print(f"HTML summary saved to {html_path}")
            continue  # Next flight

        chart_spec = load_chart_spec(args.chart)
        # --- Support both y_groups (new) and y_axes (legacy) ---
        if 'y_groups' in chart_spec:
            y_groups = chart_spec['y_groups']
            y_columns = []
            y_labels = []
            axis_indices = []
            for group in y_groups:
                axis = group.get('axis', 0)
                for series in group.get('series', []):
                    y_columns.append(series['column'])
                    y_labels.append(series['label'])
                    axis_indices.append(axis)
        elif 'y_axes' in chart_spec:
            y_axes = chart_spec['y_axes']
            y_columns = [y['column'] for y in y_axes]
            y_labels = [y['label'] for y in y_axes]
            axis_indices = [y.get('axis', i) for i, y in enumerate(y_axes)]
        else:
            raise ValueError('Chart spec must have either y_groups or y_axes')
        conn = sqlite3.connect(DB_PATH)
        flight = get_flight_segment(conn, flight_id)
        start_rowid = flight['start_rowid']
        end_rowid = flight['end_rowid']
        data, colnames = get_flight_data(conn, start_rowid, end_rowid, y_columns)
        col_idx = {name: i+1 for i, name in enumerate(y_columns)}  # +1 because rowid is at 0
        for tcol in ['zulu_year', 'zulu_mo', 'zulu_day', 'zulu_hour', 'zulu_min', 'zulu_sec']:
            col_idx[tcol] = colnames.index(tcol)
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
        # Smoothing (for plot only) - only use chart_spec['smooth']
        window_size = chart_spec.get('smooth', 0)
        if window_size and window_size > 1:
            y_data_smoothed = [moving_average_centered(yd, window_size) for yd in y_data]
            plot_y_data = y_data_smoothed
            title_suffix = f" (smoothed, window={window_size})"
        else:
            plot_y_data = y_data
            title_suffix = ""
        # Plot with support for multiple y-axes
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
            color = None
            # Try to get color from config for all charts
            if 'y_groups' in chart_spec:
                group_idx = 0
                group_series_idx = i
                count = 0
                for gidx, group in enumerate(chart_spec['y_groups']):
                    n = len(group['series'])
                    if count + n > i:
                        group_idx = gidx
                        group_series_idx = i - count
                        break
                    count += n
                series_cfg = chart_spec['y_groups'][group_idx]['series'][group_series_idx]
                color = series_cfg.get('color')
            elif 'y_axes' in chart_spec:
                if i < len(chart_spec['y_axes']):
                    color = chart_spec['y_axes'][i].get('color')
            if not color:
                # Fallback: assign a default color from matplotlib's tab10 palette
                import matplotlib
                tab_colors = matplotlib.colormaps['tab10'].colors
                color = tab_colors[i % len(tab_colors)]
            l, = ax.plot(times, y, label=y_labels[i], color=color)
            line_handles.append(l)
            ax.yaxis.label.set_color(color)
            ax.tick_params(axis='y', colors=color)
        # Set axis labels
        ax1.set_xlabel('Time since start of flight (min)')
        # Set y-labels for each axis (use first label for each axis)
        for axis in range(3):
            if axes[axis]:
                try:
                    label_idx = axis_indices.index(axis)
                    axes[axis].set_ylabel(y_labels[label_idx])
                except ValueError:
                    pass
        # Title
        title = f"Flight {flight_id}: {chart_spec.get('label', args.chart)}{title_suffix}\n{flight['start_timestamp']} to {flight['end_rowid']}"
        plt.title(title)
        # Combine all legend entries
        labs = [l.get_label() for l in line_handles]
        plt.legend(line_handles, labs, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        # Change output_path logic:
        output_path = args.output
        if not output_path or output_path.endswith(f"{args.chart}.png") or not output_path.endswith('.png'):
            output_path = os.path.join(flight_dir, f"{flight_folder}_{args.chart}.png")
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Chart saved to {output_path}")

        # CSV output path (for stats)
        if hasattr(args, 'analyze') and args.analyze:
            stats = analyze_chart_data({ycol: y_data[i] for i, ycol in enumerate(y_columns)})
            csv_path = os.path.join(flight_dir, f"{flight_folder}_{args.chart}.csv")
            print_stats_table(stats, csv_path=csv_path, flight_info=flight)
        elif args.csv:
            # If --csv is given without --analyze, still export stats
            stats = analyze_chart_data({ycol: y_data[i] for i, ycol in enumerate(y_columns)})
            csv_path = os.path.join(flight_dir, f"{flight_folder}_{args.chart}.csv")
            print_stats_table(stats, csv_path=csv_path, flight_info=flight)
        conn.close()

    # After all flights processed, if batch mode and multiple flights, create index.html in output_base
    if args.chart == "all" and len(flight_ids) > 1:
        # Gather all flight folders in output_base
        flights_info = []
        for flight_id in flight_ids:
            flight_folder = f"flight_{int(flight_id):03d}"
            flight_dir = os.path.join(output_base, flight_folder)
            summary_html = os.path.join(flight_folder, f"{flight_folder}_summary.html")
            # Try to get flight info from the first CSV or from DB
            # Use engine_performance as a likely always-present chart
            csv_path = os.path.join(flight_dir, f"{flight_folder}_engine_performance.csv")
            if not os.path.exists(csv_path):
                # fallback: any CSV
                csvs = [f for f in os.listdir(flight_dir) if f.endswith('.csv')]
                csv_path = os.path.join(flight_dir, csvs[0]) if csvs else None
            date_str = hour_str = duration_str = "?"
            year = month = None
            if csv_path and os.path.exists(csv_path):
                with open(csv_path, encoding='utf-8-sig') as f:
                    first = f.readline()
                # Try to parse flight info row
                if first.startswith('flight_id:'):
                    # Split by ',' but allow for possible missing fields
                    parts = {}
                    for s in first.strip().split(', '):
                        if ': ' in s:
                            k, v = s.split(': ', 1)
                            parts[k] = v
                    start_ts = parts.get('start_timestamp', '').replace('T', ' ').replace('Z', '')
                    end_ts = parts.get('end_timestamp', '').replace('T', ' ').replace('Z', '')
                    try:
                        dt_start = datetime.strptime(start_ts, '%Y-%m-%d %H:%M:%S')
                        dt_end = datetime.strptime(end_ts, '%Y-%m-%d %H:%M:%S')
                        date_str = dt_start.strftime('%Y-%m-%d')
                        hour_str = dt_start.strftime('%H:%M')
                        duration = dt_end - dt_start
                        duration_str = f"{duration.seconds//60} min"
                        year = dt_start.year
                        month = dt_start.month
                    except Exception:
                        year = month = None
            flights_info.append({
                'flight_id': flight_id,
                'folder': flight_folder,
                'summary_html': summary_html,
                'date': date_str,
                'hour': hour_str,
                'duration': duration_str,
                'year': year,
                'month': month
            })
        # Group by year/month
        from collections import defaultdict
        grouped = defaultdict(lambda: defaultdict(list))
        for info in flights_info:
            if info['year'] and info['month']:
                grouped[info['year']][info['month']].append(info)
        # Write index.html
        index_path = os.path.join(output_base, 'index.html')
        with open(index_path, 'w', encoding='utf-8') as idx:
            idx.write('<html><head><title>Flight Index</title><style>body{{font-family:sans-serif;}}table{{border-collapse:collapse;margin-bottom:24px;}}th,td{{border:1px solid #ccc;padding:4px;}}</style></head><body>')
            idx.write('<h1>Flight Index</h1>')
            for year in sorted(grouped.keys(), reverse=True):
                idx.write(f'<h2>{year}</h2>')
                for month in sorted(grouped[year].keys()):
                    idx.write(f'<h3>{year}-{month:02d}</h3>')
                    idx.write('<table><tr><th>Flight</th><th>Date</th><th>Hour</th><th>Duration</th><th>Report</th></tr>')
                    for info in sorted(grouped[year][month], key=lambda x: x['date']):
                        idx.write(f'<tr><td>{info["flight_id"]}</td><td>{info["date"]}</td><td>{info["hour"]}</td><td>{info["duration"]}</td><td><a href="{info["summary_html"]}">View</a></td></tr>')
                    idx.write('</table>')
            idx.write('</body></html>')
        print(f"Flight index written to {index_path}")

if __name__ == "__main__":
    main()
