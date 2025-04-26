#!/usr/bin/env python3
import re
import ast
import csv
import os
import sys

# STATIC CONFIGURATION
SQL_FILE    = "datasets/northwind-dataset/northwind_data.sql"
SQL_SCHEMA  = "datasets/northwind-dataset/schema.sql"
OUTPUT_DIR  = "datasets/northwind-dataset/csv/"

def extract_schemas(sql_file_path):
    """
    Parses CREATE TABLE statements to extract column names for each table,
    skipping constraint lines.
    """
    schemas = {}
    create_table_re = re.compile(r'^CREATE TABLE\s+(?:\w+\.)?(\w+)\s*\(', re.IGNORECASE)
    end_re          = re.compile(r'^\);')
    column_re       = re.compile(r'^\s*([A-Za-z_]\w*)\s+[\w\(\), ]+')

    current_table = None
    with open(sql_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if current_table:
                if end_re.match(stripped):
                    current_table = None
                else:
                    up = stripped.upper()
                    if up.startswith(("PRIMARY KEY", "FOREIGN KEY", "CONSTRAINT", "UNIQUE", "CHECK")) or not stripped:
                        continue
                    m = column_re.match(stripped)
                    if m:
                        schemas[current_table].append(m.group(1))
            else:
                m = create_table_re.match(stripped)
                if m:
                    table = m.group(1)
                    schemas[table] = []
                    current_table = table
    return schemas

def extract_data(sql_file_path, schemas):
    """
    Parses multi-row INSERT INTO ... VALUES (...) blocks
    and collects rows for each table, converting SQL NULL → Python None.
    """
    data = {table: [] for table in schemas}
    insert_re = re.compile(r'^INSERT INTO\s+(?:\w+\.)?(\w+)\s+VALUES', re.IGNORECASE)
    buffer = ""
    current_table = None

    with open(sql_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            m = insert_re.match(stripped)
            if m:
                current_table = m.group(1)
                buffer = ""
                # collect only what's after VALUES
                parts = line.split('VALUES', 1)[1]
                buffer += parts.strip()
                if buffer.endswith(';'):
                    buffer = buffer[:-1]
                    # sanitize SQL literals
                    buffer = buffer.replace("\\x", "")                    # remove bytea markers
                    buffer = re.sub(r'\bNULL\b', 'None', buffer)          # SQL NULL → Python None
                    try:
                        rows = ast.literal_eval(f"[{buffer}]")
                        data[current_table].extend(rows)
                    except Exception as e:
                        print(f"Warning: failed parsing {current_table}: {e}", file=sys.stderr)
                    current_table = None
                    buffer = ""
            elif current_table:
                buffer += stripped
                if stripped.endswith(';'):
                    buffer = buffer[:-1]
                    buffer = buffer.replace("\\x", "")
                    buffer = re.sub(r'\bNULL\b', 'None', buffer)
                    try:
                        rows = ast.literal_eval(f"[{buffer}]")
                        data[current_table].extend(rows)
                    except Exception as e:
                        print(f"Warning: failed parsing {current_table}: {e}", file=sys.stderr)
                    current_table = None
                    buffer = ""
    return data

def write_csvs(data, schemas, output_dir):
    """
    Writes each table's data to a CSV file using its schema for headers.
    """
    os.makedirs(output_dir, exist_ok=True)
    for table, rows in data.items():
        if not rows:
            continue
        header = schemas.get(table, [f"col{i+1}" for i in range(len(rows[0]))])
        csv_path = os.path.join(output_dir, f"{table}.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvf:
            writer = csv.writer(csvf)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {csv_path}")

def main():
    if not os.path.exists(SQL_FILE) or not os.path.exists(SQL_SCHEMA):
        print(f"Error: Missing SQL files.", file=sys.stderr)
        sys.exit(1)

    schemas = extract_schemas(SQL_SCHEMA)
    data    = extract_data(SQL_FILE, schemas)
    write_csvs(data, schemas, OUTPUT_DIR)

if __name__ == "__main__":
    main()