#!/usr/bin/env python3
"""
run_experiment_data_filler.py

Experiment using only the Intelligent Data Generator and metrics_researcher.py
to evaluate synthetic Northwind data with detailed statistical fidelity and constraint adherence analysis.

Steps:
1. Load real CSV data from 'datasets/csv/'.
2. Generate synthetic data using Intelligent Data Generator.
3. Evaluate:
   - Univariate statistical fidelity via metrics_researcher.compute_statistical_fidelity.
   - Constraint adherence via metrics_researcher.pk_uniqueness and fk_integrity.
4. Output a combined JSON report (including generation time) to 'results/'.

Requires:
    pip install intelligent-data-generator scipy pandas scikit-learn
"""

import os
import glob
import time
import json
import pandas as pd
from filling.data_generator import DataGenerator
from parsing.parsing import parse_create_tables
import metrics

# Configuration
CSV_DIR     = 'datasets/csv/'
SCHEMA_FILE = 'datasets/northwind-dataset/schema.sql'
RESULTS_DIR = 'results/'

def load_real_data():
    """Load each CSV in CSV_DIR into a pandas DataFrame."""
    real_data = {}
    for path in glob.glob(os.path.join(CSV_DIR, '*.csv')):
        table = os.path.splitext(os.path.basename(path))[0]
        real_data[table] = pd.read_csv(path)
    return real_data

def setup_data_filler(schema_path, real_data):
    """Initialize DataGenerator from DDL and desired row counts."""
    ddl = open(schema_path, 'r').read()
    tables = parse_create_tables(ddl)
    num_rows = {t: len(df) for t, df in real_data.items()}
    dg = DataGenerator(
        tables,
        num_rows_per_table=num_rows,
        max_attepts_to_generate_value=100,

        # guess_column_type_mappings=True,
        # threshold_for_guessing=95
    )
    return dg, tables

def generate_data_filler(dg):
    """
    Generate synthetic data and measure time; convert each table's
    list-of-dicts into a pandas DataFrame.
    """
    start = time.time()
    raw = dg.generate_data()  # { table_name: [ {col: val}, ... ], ... }
    duration = time.time() - start
    synth_data = {table: pd.DataFrame(rows) for table, rows in raw.items()}
    return synth_data, duration

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load real data
    real_data = load_real_data()

    # 2. Generate synthetic data
    dg, schema_tables = setup_data_filler(SCHEMA_FILE, real_data)
    synth_data, gen_time = generate_data_filler(dg)

    # 3. Statistical fidelity (univariate)
    fidelity = {}
    for table, real_df in real_data.items():
        synth_df = synth_data.get(table)
        if synth_df is not pd.DataFrame.empty:
            continue
        col_stats = {}
        for col in real_df.columns:
            col_stats[col] = metrics.compute_statistical_fidelity(real_df[col], synth_df[col])
        fidelity[table] = col_stats

    # 4. Constraint adherence
    constraints = {}
    for table, df in synth_data.items():
        meta = schema_tables[table]  # {'columns': [...], 'primary_key': [...], 'foreign_keys': [...], ...}

        # Primary key uniqueness
        pk_cols = meta['primary_key']
        constraints.setdefault(table, {})['pk_uniqueness'] = metrics.pk_uniqueness(df, pk_cols)

        # Foreign key integrity
        for fk in meta.get('foreign_keys', []):
            fk_col        = fk['column']               # e.g. 'category_id'
            parent_table  = fk['referenced_table']     # e.g. 'categories'
            parent_column = fk['referenced_column']    # e.g. 'category_id'
            constraints[table][f'fk_{fk_col}'] = metrics.fk_integrity(
                df, synth_data[parent_table], fk_col, parent_column
            )

    # 5. Compile and save results
    results = {
        'generation_time_seconds': gen_time,
        'statistical_fidelity': fidelity,
        'constraint_adherence': constraints
    }

    # 6. Interpret and display human-readable analysis
    metrics.interpret_results({
        'statistical': results['statistical_fidelity'],
        'constraints': results['constraint_adherence']
    })

    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Done: synthetic data generated in {gen_time:.2f}s; results saved to {RESULTS_DIR}/results.json")

if __name__ == '__main__':
    main()