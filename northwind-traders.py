# run_experiment_data_filler.py

# !/usr/bin/env python3
"""
Experiment using only the Intelligent Data Generator and metrics_researcher.py
to evaluate synthetic Northwind data via CSV files.

Steps:
1. Load real CSV data from 'datasets/csv/'.
2. Generate synthetic CSVs via DataGenerator.export_data_files().
3. Load both CSV dirs and compute all metrics with evaluate_from_csv_dirs().
4. Interpret and save results.
"""

import os
import time
import json
from filling.data_generator import DataGenerator
from parsing.parsing import parse_create_tables
import metrics

# Configuration
CSV_DIR = 'datasets/csv/'
SYNTH_CSV_DIR = 'data_generator_results/'
SCHEMA_FILE = 'datasets/northwind-dataset/schema.sql'
RESULTS_DIR = 'results/'


def setup_data_generator(schema_path, real_dir):
    # parse schema
    ddl = open(schema_path).read()
    schema_tables = parse_create_tables(ddl)
    # count rows in each real CSV
    import glob, pandas as pd, os
    num_rows = {
        os.path.splitext(os.path.basename(p))[0]: len(pd.read_csv(p))
        for p in glob.glob(os.path.join(real_dir, '*.csv'))
    }
    dg = DataGenerator(
        schema_tables,
        num_rows_per_table=num_rows,
        max_attepts_to_generate_value=100,
        guess_column_type_mappings=True,
        threshold_for_guessing=96
    )
    return dg, schema_tables


def generate_and_export(dg):
    start = time.time()
    dg.generate_data()  # in-memory, but we only care about CSV export
    dg.export_data_files(output_dir=SYNTH_CSV_DIR, file_type='CSV')
    return time.time() - start


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(SYNTH_CSV_DIR, exist_ok=True)

    # 1 & 2: setup & export synthetic CSVs
    dg, schema_tables = setup_data_generator(SCHEMA_FILE, CSV_DIR)
    gen_time = generate_and_export(dg)

    # 3: evaluate from CSV dirs
    full_results = metrics.evaluate_from_csv_dirs(
        real_dir=CSV_DIR,
        synth_dir=SYNTH_CSV_DIR,
        schema_tables=schema_tables
    )
    full_results['generation_time_s'] = gen_time

    # 4: interpret and save
    # In your run_experiment_data_filler.py, replace the interpret_full_results(...) call with:
    metrics.interpret_and_save(full_results, RESULTS_DIR)
    # Save results to CSV


    print(f"\nCompleted in {gen_time:.2f}s; results saved to {RESULTS_DIR}/results.json")


if __name__ == '__main__':
    main()
