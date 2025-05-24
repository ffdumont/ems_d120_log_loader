# ems_d120_log_loader

A command-line tool to convert Dynon EMS-D120 `.csv` log files into a structured SQLite database.  
Supports label-based schema inference and data cleaning (e.g. decimal conversion, missing values).

## Features

- Auto-detects schema from Dynon `Label` row
- Converts `N/A` values to NULL
- Supports EMS-D120 logs (and compatible variants)
- Handles decimal comma conversion
- Outputs a ready-to-query `.sqlite` database

## Installation

```bash
git clone https://github.com/youruser/dynon_csv_to_sqlite.git
cd dynon_csv_to_sqlite
pip install -r requirements.txt
