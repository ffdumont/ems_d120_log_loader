import argparse
import csv
import glob
import hashlib
import logging
import os
import sqlite3
import sys
from typing import List, Tuple, Dict, Any

# --- Logging Setup ---
def setup_logging(log_path: str, verbose: bool):
    logger = logging.getLogger("dynon_import")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    if verbose:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

# --- CSV Parsing and Normalization ---
def find_label_row(csv_path: str) -> Tuple[int, List[str]]:
    with open(csv_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            # Strip quotes and whitespace, then check for Label
            first_cell = line.strip().split(";")[0].strip().strip('"')
            if first_cell == "Label":  # Label row
                columns = [c.strip().strip('"') for c in line.strip().split(";")]
                # Remove trailing empty columns
                while columns and columns[-1] == "":
                    columns.pop()
                return idx, columns
    raise ValueError(f"No 'Label' row found in {csv_path}")

def clean_column_name(name: str) -> str:
    # Remove or rename duplicate columns by appending an index
    base = name.strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_")
    return base

def make_unique_columns(columns: list) -> list:
    seen = {}
    result = []
    for col in columns:
        base = clean_column_name(col)
        if base not in seen:
            seen[base] = 1
            result.append(base)
        else:
            seen[base] += 1
            result.append(f"{base}_{seen[base]}")
    return result

def infer_column_types(rows: List[List[str]]) -> List[str]:
    types = []
    for col in zip(*rows):
        col_types = set()
        for val in col:
            v = val.replace(",", ".") if val else val
            if v in ("", "N/A", None):
                continue
            try:
                int(float(v))
                col_types.add("INTEGER")
            except Exception:
                try:
                    float(v)
                    col_types.add("REAL")
                except Exception:
                    col_types.add("TEXT")
        if not col_types:
            types.append("TEXT")
        elif "TEXT" in col_types:
            types.append("TEXT")
        elif "REAL" in col_types:
            types.append("REAL")
        else:
            types.append("INTEGER")
    return types

def normalize_row(row: List[str]) -> List[Any]:
    norm = []
    for v in row:
        if v == "N/A":
            norm.append(None)
        else:
            v2 = v.replace(",", ".")
            norm.append(v2)
    return norm

def compute_file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def compute_row_hash(row: List[Any]) -> str:
    joined = "|".join(["" if v is None else str(v) for v in row])
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()

# --- SQLite Operations ---
def create_table(conn, table: str, columns: List[str], types: List[str], overwrite: bool, logger):
    cur = conn.cursor()
    # Removed redundant drop logic; handled in main()
    col_defs = ", ".join([f'"{c}" {t}' for c, t in zip(columns, types)])
    sql = f"""
    CREATE TABLE IF NOT EXISTS {table} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        {col_defs},
        file_name TEXT,
        file_hash TEXT,
        row_hash TEXT UNIQUE
    )"""
    logger.debug(f"Creating table with SQL: {sql}")
    cur.execute(sql)
    conn.commit()

def import_csv(
    csv_path: str,
    conn,
    table: str,
    logger,
    file_hashes: set,
    row_hashes: set
) -> Tuple[int, int]:
    try:
        file_hash = compute_file_hash(csv_path)
    except UnicodeDecodeError as e:
        logger.error(f"File encoding error in {csv_path}: {e}")
        return 0, 0
    if file_hash in file_hashes:
        logger.info(f"Skipping {csv_path}: file already imported (hash match)")
        return 0, 0
    try:
        label_idx, columns = find_label_row(csv_path)
    except Exception as e:
        logger.error(f"Header parse error in {csv_path}: {e}")
        return 0, 0
    # Remove trailing empty columns from header
    while columns and columns[-1] == "":
        columns.pop()
    columns_clean = make_unique_columns(columns)
    data_rows = []
    try:
        with open(csv_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i <= label_idx:
                    continue
                row = [v.strip() for v in line.strip().split(";")]
                # Remove trailing empty columns to match header
                while len(row) > 0 and row[-1] == "":
                    row.pop()
                # Only compare up to the number of header columns
                if len(row) < len(columns):
                    logger.warning(f"Row length mismatch in {csv_path} at line {i+1}")
                    continue
                # If row is longer, trim to match header
                if len(row) > len(columns):
                    row = row[:len(columns)]
                data_rows.append(row)
    except UnicodeDecodeError as e:
        logger.error(f"File encoding error in {csv_path}: {e}")
        return 0, 0
    if not data_rows:
        logger.warning(f"No data rows found in {csv_path}")
        return 0, 0
    types = infer_column_types(data_rows)
    create_table(conn, table, columns_clean, types, False, logger)
    cur = conn.cursor()
    insert_sql = f"INSERT OR IGNORE INTO {table} (" + ", ".join([f'\"{c}\"' for c in columns_clean]) + ", file_name, file_hash, row_hash) VALUES (" + ", ".join(["?" for _ in columns_clean]) + ", ?, ?, ?)"
    inserted = 0
    skipped = 0
    for row in data_rows:
        norm = normalize_row(row)
        row_hash = compute_row_hash(norm)
        if row_hash in row_hashes:
            skipped += 1
            continue
        try:
            cur.execute(insert_sql, [*norm, os.path.basename(csv_path), file_hash, row_hash])
            inserted += 1
            row_hashes.add(row_hash)
        except Exception as e:
            logger.error(f"Insert error in {csv_path}, row={norm}: {e}")
    conn.commit()
    file_hashes.add(file_hash)
    logger.info(f"Imported {inserted} rows from {csv_path} ({skipped} duplicates)")
    return inserted, skipped

def get_existing_hashes(conn, table: str) -> Tuple[set, set]:
    cur = conn.cursor()
    file_hashes = set()
    row_hashes = set()
    try:
        cur.execute(f"SELECT DISTINCT file_hash FROM {table}")
        file_hashes = {r[0] for r in cur.fetchall() if r[0]}
        cur.execute(f"SELECT row_hash FROM {table}")
        row_hashes = {r[0] for r in cur.fetchall() if r[0]}
    except Exception:
        pass
    return file_hashes, row_hashes

# --- Main CLI ---
def main():
    parser = argparse.ArgumentParser(description="Import Dynon EMS-D120 CSV logs into SQLite.")
    parser.add_argument("--input", required=True, nargs="+", help="Input CSV file(s), supports wildcards")
    parser.add_argument("--output", required=True, help="Output SQLite database file")
    parser.add_argument("--table", default="flight_log", help="Table name (default: flight_log)")
    parser.add_argument("--log", default="import.log", help="Log file path (default: import.log)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose console logging")
    parser.add_argument("--overwrite", action="store_true", help="Drop and recreate table before import")
    args = parser.parse_args()

    logger = setup_logging(args.log, args.verbose)
    logger.info(f"Starting import: input={args.input}, output={args.output}, table={args.table}")
    files = []
    for pattern in args.input:
        files.extend(glob.glob(pattern))
    if not files:
        logger.error("No input files found.")
        sys.exit(1)
    conn = sqlite3.connect(args.output)
    if args.overwrite:
        cur = conn.cursor()
        logger.info(f"Dropping table {args.table} (overwrite enabled)")
        cur.execute(f"DROP TABLE IF EXISTS {args.table}")
        conn.commit()
    file_hashes, row_hashes = get_existing_hashes(conn, args.table)
    total_inserted = 0
    total_skipped = 0
    for f in files:
        logger.info(f"Processing file: {f}")
        try:
            inserted, skipped = import_csv(f, conn, args.table, logger, file_hashes, row_hashes)
            total_inserted += inserted
            total_skipped += skipped
        except Exception as e:
            logger.error(f"Failed to import {f}: {e}")
    logger.info(f"Import complete. Total inserted: {total_inserted}, total skipped: {total_skipped}")
    conn.close()

if __name__ == "__main__":
    main()
