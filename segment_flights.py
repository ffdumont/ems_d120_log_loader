import sqlite3
from datetime import datetime

def segment_flights(sqlite_path, table_name):
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    # Drop and recreate flight_segments table
    cur.execute("DROP TABLE IF EXISTS flight_segments")
    cur.execute("""
        CREATE TABLE flight_segments (
            flight_id INTEGER,
            start_timestamp TEXT,
            end_timestamp TEXT,
            duration_seconds INTEGER,
            start_rowid INTEGER,
            end_rowid INTEGER,
            record_count INTEGER
        )
    """)
    # Read all timestamps and rowids, strictly ordered by timestamp
    cur.execute(f"SELECT rowid, zulu_year, zulu_mo, zulu_day, zulu_hour, zulu_min, zulu_sec FROM {table_name} WHERE zulu_year IS NOT NULL AND zulu_mo IS NOT NULL AND zulu_day IS NOT NULL AND zulu_hour IS NOT NULL AND zulu_min IS NOT NULL AND zulu_sec IS NOT NULL")
    rows = []
    for row in cur.fetchall():
        rowid, y, mo, d, h, mi, s = row
        try:
            y_int = int(y)
            if y_int < 100:
                y_int += 2000
            t = datetime.strptime(f"{y_int:04d}-{int(mo):02d}-{int(d):02d}T{int(h):02d}:{int(mi):02d}:{int(s):02d}", "%Y-%m-%dT%H:%M:%S")
            rows.append((rowid, t))
        except Exception:
            continue  # skip invalid timestamps
    # Sort all rows strictly by timestamp
    rows.sort(key=lambda x: x[1])
    flights = []
    flight_rows = []
    prev_time = None
    for rowid, t in rows:
        if prev_time is not None:
            delta = (t - prev_time).total_seconds()
            if delta > 600:
                if flight_rows:
                    flights.append(flight_rows)
                flight_rows = []
        flight_rows.append((rowid, t))
        prev_time = t
    if flight_rows:
        flights.append(flight_rows)
    # Write flight_segments
    for i, flight in enumerate(flights):
        flight_id = i + 1
        start_rowid = flight[0][0]
        end_rowid = flight[-1][0]
        start_ts = flight[0][1].strftime("%Y-%m-%dT%H:%M:%S")
        end_ts = flight[-1][1].strftime("%Y-%m-%dT%H:%M:%S")
        duration = int((flight[-1][1] - flight[0][1]).total_seconds()) if len(flight) > 1 else 0
        record_count = len(flight)
        # Filter out noise/invalid flights
        if duration < 60 or record_count < 3:
            print(f"Skipping segment: flight_id={flight_id}, duration={duration}s, records={record_count}")
            continue
        if duration < 0:
            print(f"Warning: Skipping flight_id={flight_id} with negative duration: {start_ts} to {end_ts}")
            continue
        cur.execute(
            "INSERT INTO flight_segments (flight_id, start_timestamp, end_timestamp, duration_seconds, start_rowid, end_rowid, record_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (flight_id, start_ts, end_ts, duration, start_rowid, end_rowid, record_count)
        )
    conn.commit()
    conn.close()
    print(f"Segmented {len([f for f in flights if (len(f) > 2 and (f[-1][1] - f[0][1]).total_seconds() >= 60)])} flights into 'flight_segments' table.")

# Example usage:
# segment_flights('logs_test.sqlite', 'flight_log')

if __name__ == "__main__":
    # For direct script execution, segment flights
    segment_flights('logs_test.sqlite', 'flight_log')
