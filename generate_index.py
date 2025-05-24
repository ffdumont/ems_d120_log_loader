import os
import csv
from datetime import datetime
from collections import defaultdict

CHARTS_OUTPUT = "charts_output"
SUMMARY_SUFFIX = "_summary.html"


def parse_flight_info_from_csv(csv_path):
    """Parse the first line of a CSV to extract flight info (date, hour, duration, year, month)."""
    date_str = hour_str = duration_str = "?"
    year = month = None
    if not os.path.exists(csv_path):
        return date_str, hour_str, duration_str, year, month
    with open(csv_path, encoding='utf-8-sig') as f:
        first = f.readline()
    if first.startswith('flight_id:'):
        parts = {}
        for s in first.strip().split(';'):
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
    return date_str, hour_str, duration_str, year, month


def main():
    flights_info = []
    for folder in sorted(os.listdir(CHARTS_OUTPUT)):
        flight_dir = os.path.join(CHARTS_OUTPUT, folder)
        if not os.path.isdir(flight_dir):
            continue
        if not folder.startswith("flight_"):
            continue
        summary_html = os.path.join(folder, f"{folder}_summary.html")
        # Find a CSV file to extract info
        csvs = [f for f in os.listdir(flight_dir) if f.endswith('.csv')]
        if not csvs:
            continue
        csv_path = os.path.join(flight_dir, csvs[0])
        date_str, hour_str, duration_str, year, month = parse_flight_info_from_csv(csv_path)
        flight_id = folder.replace("flight_", "")
        flights_info.append({
            'flight_id': flight_id,
            'folder': folder,
            'summary_html': summary_html,
            'date': date_str,
            'hour': hour_str,
            'duration': duration_str,
            'year': year,
            'month': month
        })
    # Group by year/month
    grouped = defaultdict(lambda: defaultdict(list))
    for info in flights_info:
        if info['year'] and info['month']:
            grouped[info['year']][info['month']].append(info)
    # Write index.html
    index_path = os.path.join(CHARTS_OUTPUT, 'index.html')
    with open(index_path, 'w', encoding='utf-8') as idx:
        idx.write('<html><head><title>Flight Index</title><style>body{font-family:sans-serif;}table{border-collapse:collapse;margin-bottom:24px;}th,td{border:1px solid #ccc;padding:4px;}</style></head><body>')
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
