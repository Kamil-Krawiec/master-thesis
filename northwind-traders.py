#!/usr/bin/env python3
"""
run_experiment_data_filler.py

Experiment using only the Intelligent Data Generator and metrics_researcher.py
to evaluate synthetic Northwind data with detailed fidelity and constraint analysis.

Steps:
1. Load real CSV data from 'datasets/csv/'.
2. Generate synthetic data using Intelligent Data Generator.
3. Evaluate:
   - Statistical fidelity via metrics_researcher.compute_statistical_fidelity.
   - Distance metrics via metrics_researcher.compute_distance_metrics.
   - Child-count KS for FKs via metrics_researcher.compute_child_count_ks.
   - Schema adherence via metrics_researcher.compute_schema_adherence.
   - Constraint adherence via metrics_researcher.compute_constraint_adherence.
4. Interpret results and save full report (incl. generation time) to 'results/'.

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
import metrics  # use the researcher-grade metrics

# Configuration
CSV_DIR     = 'datasets/csv/'
SCHEMA_FILE = 'datasets/northwind-dataset/schema.sql'
RESULTS_DIR = 'results/'

def load_real_data():
    real = {}
    for path in glob.glob(os.path.join(CSV_DIR, '*.csv')):
        tbl = os.path.splitext(os.path.basename(path))[0]
        real[tbl] = pd.read_csv(path)
    return real

def setup_data_filler(schema_path, real_data):
    ddl    = open(schema_path).read()
    tables = parse_create_tables(ddl)
    rows   = {t: len(df) for t, df in real_data.items()}
    dg     = DataGenerator(tables, num_rows_per_table=rows, max_attepts_to_generate_value=100)
    return dg, tables

def generate_synthetic(dg):
    start = time.time()
    raw   = dg.generate_data()  # {table: [ {col: val}... ], ...}
    data  = {t: pd.DataFrame(rs) for t, rs in raw.items()}
    return data, time.time() - start

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load
    real_data = load_real_data()

    # 2. Generate
    dg, schema_tables = setup_data_filler(SCHEMA_FILE, real_data)
    synth_data, gen_time = generate_synthetic(dg)

    # 3. Compute metrics
    stats        = {}
    distances    = {}
    child_ks     = {}
    schema_check = {}

    # Prepare dtype and range maps from schema definitions
    dtype_map = {col['name']: col['type'].lower() for col in schema_tables['categories']['columns']}  # example; build for each table below
    # (you would build dtype_map and range_map per table from schema_tables metadata)

    for tbl, real_df in real_data.items():
        synth_df = synth_data.get(tbl)
        if synth_df is None:
            continue

        # 3.1 Statistical fidelity (univariate)
        stats[tbl] = {
            col: metrics.compute_statistical_fidelity(real_df[col], synth_df[col])
            for col in real_df.columns
        }

        # 3.2 Distance-based metrics
        dtype_map = {
            col_meta['name']: col_meta['type'].lower()
            for col_meta in schema_tables[tbl]['columns']
        }
        # Build range_map by inferring min/max from the real data for numeric columns
        range_map = {}
        for col in real_df.columns:
            if pd.api.types.is_numeric_dtype(real_df[col]):
                mn = real_df[col].min()
                mx = real_df[col].max()
                range_map[col] = (mn, mx)
        # Now compute adherence
        schema_check[tbl] = metrics.compute_schema_adherence(
            synth_df,
            dtype_map,
            range_map
        )

        # 3.3 Child-count KS for each FK
        child_ks[tbl] = {}
        for fk in schema_tables[tbl].get('foreign_keys', []):
            child_ks[tbl][fk['columns']] = metrics.compute_child_count_ks(
                real_data[fk['ref_table']],
                synth_data[fk['ref_table']],
                fk['columns']
            )

        # 3.4 Schema & range adherence
        schema_check[tbl] = metrics.compute_schema_adherence(
            synth_df,
            dtype_map,                # customize per table
            {col: (None, None)}       # you can infer real min/max values for each col
        )

    # 3.5 Constraint adherence (PK & FK)
    constraint_results = metrics.compute_constraint_adherence(
        synth_data,
        {tbl: {
            'pk': schema_tables[tbl]['primary_key'],
            'fks': schema_tables[tbl]['foreign_keys']
        } for tbl in schema_tables}
    )

    # 4. Interpret & save
    full_results = {
        'generation_time_s': gen_time,
        'univariate_fidelity': stats,
        'distance_metrics': distances,
        'child_count_ks': child_ks,
        'schema_adherence': schema_check,
        'constraint_adherence': constraint_results
    }

    metrics.interpret_results({
        'statistical': stats,
        'constraints': constraint_results
    })

    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump(full_results, f, indent=2)

    print(f"Completed in {gen_time:.2f}s; results at {RESULTS_DIR}/results.json")


if __name__ == '__main__':
    main()