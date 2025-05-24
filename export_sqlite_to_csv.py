import sqlite3
import csv
import json
from datetime import datetime

# Hardcoded paths (can be parameterized later)
DB_PATH = 'logs_test.sqlite'
CONFIG_PATH = 'export_config.json'
CSV_PATH = 'exported_data.csv'
TABLE = 'flight_log'

# Load config
with open(CONFIG_PATH, encoding='utf-8') as f:
    config = json.load(f)
fields = config['output_fields']

# Prepare output headers and db columns
headers = [f['label'] for f in fields]
db_columns = [f['column'] for f in fields]

# Build SQL SELECT
select_exprs = []
for col in db_columns:
    if col == 'timestamp':
        # Compose ISO timestamp from zulu fields
        select_exprs.append(
            "printf('%04d-%02d-%02dT%02d:%02d:%02d', "
            "CAST(zulu_year AS INTEGER), "
            "CAST(zulu_mo AS INTEGER), "
            "CAST(zulu_day AS INTEGER), "
            "CAST(zulu_hour AS INTEGER), "
            "CAST(zulu_min AS INTEGER), "
            "CAST(zulu_sec AS INTEGER)) AS timestamp"
        )
    else:
        select_exprs.append(f'"{col}"')
select_sql = f"SELECT {', '.join(select_exprs)} FROM {TABLE}"

# Query DB
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute(select_sql)
rows = cursor.fetchall()

# Write CSV
with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter=';', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(headers)
    row_count = 0
    for idx, row in enumerate(rows, 1):
        # Remove rows with all fields empty or with timestamp '0000-00-00T00:00:00'
        if len(row) > 0 and row[0] == '0000-00-00T00:00:00':
            continue
        # Convert decimals: replace '.' with ',' for numeric (non-timestamp) fields
        out_row = []
        for i, v in enumerate(row):
            if v is None:
                out_row.append('')
            elif i > 0 and isinstance(v, str) and v.replace('.', '', 1).isdigit():
                out_row.append(v.replace('.', ','))
            else:
                out_row.append(v)
        writer.writerow(out_row)
        row_count += 1
        if row_count % 1000 == 0:
            print(f"Exported {row_count} rows...")

print(f"Exported {row_count} rows to {CSV_PATH}")

conn.close()
