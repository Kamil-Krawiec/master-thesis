#!/usr/bin/env python3
"""
run_experiment_data_filler_refactored.py

Experiment comparing Intelligent Data Generator and Mockaroo outputs
against real Northwind CSVs using the refactored metrics evaluator.
"""
import os
import time
import glob
import pandas as pd
from filling import DataGenerator
from parsing import parse_create_tables
from metrics import evaluate_from_csv_dirs

# Configuration
REAL_CSV_DIR     = 'datasets/csv/'
SYNTH_CSV_DIR    = 'data_generator_data/'
MOCKAROO_CSV_DIR = 'mockaroo_fabricate/'
SCHEMA_FILE      = 'datasets/northwind-dataset/schema.sql'
RESULTS_DIR      = 'results/'


def setup_data_generator(schema_path: str, real_dir: str):
    """
    Prepare DataGenerator with schema and real row counts.
    Returns (DataGenerator, schema_tables).
    """
    ddl = open(schema_path).read()
    schema_tables = parse_create_tables(ddl)
    num_rows = {
        os.path.splitext(os.path.basename(path))[0]: len(pd.read_csv(path))
        for path in glob.glob(os.path.join(real_dir, '*.csv'))
    }
    dg = DataGenerator(
        schema_tables,
        num_rows=num_rows,
        max_attepts_to_generate_value=100,
        guess_column_type_mappings=True,
        threshold_for_guessing=96
    )
    return dg, schema_tables


def generate_and_export(dg: DataGenerator, out_dir: str) -> float:
    """
    Generate synthetic data and export CSVs to out_dir. Returns generation time in seconds.
    """
    start = time.time()
    dg.generate_data()
    dg.export_data_files(output_dir=out_dir, file_type='CSV')
    return time.time() - start


def main():
    # ensure directories exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(SYNTH_CSV_DIR, exist_ok=True)
    os.makedirs(MOCKAROO_CSV_DIR, exist_ok=True)

    # 1) Prepare and export DataGenerator CSVs
    dg, schema_tables = setup_data_generator(SCHEMA_FILE, REAL_CSV_DIR)
    gen_time = generate_and_export(dg, SYNTH_CSV_DIR)
    print(f"[DataGenerator] generation time: {gen_time:.2f} seconds")

    # 2) Evaluate DataGenerator outputs
    evaluate_from_csv_dirs(
        real_dir=REAL_CSV_DIR,
        synth_dir=SYNTH_CSV_DIR,
        schema_tables=schema_tables,
        output_dir=os.path.join(RESULTS_DIR, 'data_generator')
    )

    # 3) Evaluate Mockaroo outputs
    evaluate_from_csv_dirs(
        real_dir=REAL_CSV_DIR,
        synth_dir=MOCKAROO_CSV_DIR,
        schema_tables=schema_tables,
        output_dir=os.path.join(RESULTS_DIR, 'mockaroo')
    )

    print(f"Completed: reports saved under {RESULTS_DIR}")


if __name__ == '__main__':
    main()
