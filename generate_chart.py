import argparse
import json
import sqlite3
import matplotlib.pyplot as plt
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
    for i, y in enumerate(y_data):
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
    title = f"Flight {args.flight}: {chart_spec.get('label', args.chart)}\n{flight['start_timestamp']} to {flight['end_timestamp']}"
    plt.title(title)
    # Combine all legend entries
    labs = [l.get_label() for l in line_handles]
    plt.legend(line_handles, labs, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    if args.output:
        plt.savefig(args.output, bbox_inches='tight')
        print(f"Chart saved to {args.output}")
    else:
        plt.show()
    conn.close()

if __name__ == "__main__":
    main()
