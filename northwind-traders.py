#!/usr/bin/env python3
"""
run_experiment_data_filler.py

Experiment comparing Intelligent Data Generator and Mockaroo outputs
against real Northwind CSVs using metrics_researcher.py.
"""

import os
import time
from filling.data_generator import DataGenerator
from parsing.parsing import parse_create_tables
import metrics

# Configuration
REAL_CSV_DIR     = 'datasets/csv/'
SYNTH_CSV_DIR    = 'data_generator_data/'
MOCKAROO_CSV_DIR = 'mocaroo_fabricate/'
SCHEMA_FILE      = 'datasets/northwind-dataset/schema.sql'
RESULTS_DIR      = 'results/'

def setup_data_generator(schema_path, real_dir):
    """Prepare DataGenerator with schema and real row counts."""
    ddl = open(schema_path).read()
    schema_tables = parse_create_tables(ddl)
    import glob, pandas as pd, os
    num_rows = {
        os.path.splitext(os.path.basename(p))[0]: len(pd.read_csv(p))
        for p in glob.glob(os.path.join(real_dir, '*.csv'))
    }
    dg = DataGenerator(
        schema_tables,
        num_rows=100,
        max_attepts_to_generate_value=100,
        guess_column_type_mappings=True,
        threshold_for_guessing=96
    )
    return dg, schema_tables

def generate_and_export(dg, out_dir):
    """Generate in-memory and export synthetic CSVs to out_dir."""
    start = time.time()
    dg.generate_data()  # fill in-memory
    dg.export_data_files(output_dir=out_dir, file_type='CSV')
    return time.time() - start

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(SYNTH_CSV_DIR, exist_ok=True)

    # 1) Prepare and export DataGenerator CSVs
    dg, schema_tables = setup_data_generator(SCHEMA_FILE, REAL_CSV_DIR)
    gen_time = generate_and_export(dg, SYNTH_CSV_DIR)

    # 2) Evaluate DataGenerator vs real
    dg_results = metrics.evaluate_from_csv_dirs(
        real_dir=REAL_CSV_DIR,
        synth_dir=SYNTH_CSV_DIR,
        schema_tables=schema_tables
    )
    metrics.interpret_and_save(dg_results, os.path.join(RESULTS_DIR, 'data_generator'))

    # 3) Evaluate Mockaroo vs real
    mr_results = metrics.evaluate_from_csv_dirs(
        real_dir=REAL_CSV_DIR,
        synth_dir=MOCKAROO_CSV_DIR,
        schema_tables=schema_tables
    )
    metrics.interpret_and_save(mr_results, os.path.join(RESULTS_DIR, 'mockaroo'))

    print(f"Completed: reports saved under {RESULTS_DIR}/data_generator and {RESULTS_DIR}/mockaroo")

if __name__ == '__main__':
    main()