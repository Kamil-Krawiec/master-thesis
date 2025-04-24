#!/usr/bin/env python3
"""
run_experiment_data_filler.py

Experiment using only the Intelligent Data Generator and metrics.py
to evaluate synthetic Northwind data.

Steps:
1. Load real CSV data from 'datasets/csv/'.
2. Generate synthetic data using Intelligent Data Generator.
3. Evaluate:
   - Statistical fidelity (KS test, JS divergence).
   - Constraint adherence (PK uniqueness, FK integrity).
4. Output results to 'results/' directory.

Requires:
    pip install intelligent-data-generator scipy pandas
"""

import os
import glob
import time
import json
import pandas as pd
from filling.data_generator import DataGenerator
from parsing.parsing import parse_create_tables
import metrics  # assumes metrics.py is in the same directory

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
    """Initialize the Intelligent Data Generator from DDL and row counts."""
    ddl = open(schema_path, 'r').read()
    tables = parse_create_tables(ddl)
    num_rows = {t: len(df) for t, df in real_data.items()}
    dg = DataGenerator(
        tables,
        num_rows_per_table=num_rows,
        guess_column_type_mappings=True,
        threshold_for_guessing=95
    )
    return dg

def generate_data_filler(dg):
    """
    Generate synthetic data and measure time; convert each table's list-of-dicts
    into a pandas DataFrame.
    """
    import time
    import pandas as pd

    start = time.time()
    raw = dg.generate_data()  # { table_name: [ {col: val, …}, … ], … }
    duration = time.time() - start

    # Convert each list of dicts into a DataFrame
    gen_data = {
        table: pd.DataFrame(rows)
        for table, rows in raw.items()
    }

    return gen_data, duration

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load real data
    real_data = load_real_data()

    # 2. Generate synthetic data
    dg = setup_data_filler(SCHEMA_FILE, real_data)
    synth_data, gen_time = generate_data_filler(dg)

    # 3. Evaluation
    stats = {}
    for table, real_df in real_data.items():
        synth_df = synth_data.get(table)
        if synth_df is None:
            continue
        # statistical fidelity
        col_stats = {}
        for col in real_df.columns:
            if pd.api.types.is_numeric_dtype(real_df[col]):
                col_stats[col] = metrics.univariate_ks(real_df[col], synth_df[col])
            else:
                col_stats[col] = metrics.categorical_chi2(real_df[col], synth_df[col])
        stats[table] = col_stats

    # constraint adherence
    # parse schema to extract PK/FK info
    ddl = open(SCHEMA_FILE, 'r').read()
    tables = parse_create_tables(ddl)
    schema_info = tables  # tables includes PK/FK metadata in DataGenerator schema objects
    constr = {}
    for table, synth_df in synth_data.items():
        pk = tables[table].primary_key
        fk_list = tables[table].foreign_keys
        # PK uniqueness
        constr.setdefault(table, {})['pk_uniqueness'] = metrics.pk_uniqueness(synth_df, pk)
        # FK integrity
        for fk in fk_list:
            parent = fk.parent_table
            constr[table][f"fk_{fk.child_column}"] = metrics.fk_integrity(
                synth_df, synth_data[parent], fk.child_column, fk.parent_primary_key
            )

    # 4. Save results
    results = {
        'generation_time': gen_time,
        'statistical_fidelity': stats,
        'constraint_adherence': constr
    }
    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("Experiment completed. Results saved to results/results.json")

if __name__ == '__main__':
    main()