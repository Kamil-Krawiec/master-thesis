# run_experiment_data_filler.py

#!/usr/bin/env python3
"""
Experiment using only the Intelligent Data Generator and metrics_researcher.py
to evaluate synthetic Northwind data with detailed fidelity and constraint analysis.
"""

import os
import glob
import time
import json
import pandas as pd
from filling.data_generator import DataGenerator
from parsing.parsing import parse_create_tables
import metrics

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
    ddl           = open(schema_path).read()
    schema_tables = parse_create_tables(ddl)
    num_rows      = {t: len(df) for t, df in real_data.items()}
    dg = DataGenerator(schema_tables,
                       num_rows_per_table=num_rows,
                       max_attepts_to_generate_value=100)
    return dg, schema_tables

def generate_synthetic(dg):
    start = time.time()
    raw   = dg.generate_data()
    synth = {t: pd.DataFrame(rows) for t, rows in raw.items()}
    return synth, time.time() - start

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    real_data    = load_real_data()
    dg, schema_tables = setup_data_filler(SCHEMA_FILE, real_data)
    synth_data, gen_time = generate_synthetic(dg)

    stats        = {}
    distances    = {}
    child_ks     = {}
    schema_check = {}

    for tbl, real_df in real_data.items():
        synth_df = synth_data.get(tbl)
        if synth_df.empty:
            continue

        stats[tbl] = {
            col: metrics.compute_statistical_fidelity(real_df[col], synth_df[col])
            for col in real_df.columns
        }

        distances[tbl] = {
            col: metrics.compute_distance_metrics(real_df[col], synth_df[col])
            for col in real_df.columns
        }

        child_ks[tbl] = {}
        for fk in schema_tables[tbl].get('foreign_keys', []):
            fk_cols = fk['columns']
            key     = tuple(fk_cols) if isinstance(fk_cols, list) else fk_cols
            child_ks[tbl][key] = metrics.compute_child_count_ks(
                real_data[fk['ref_table']],
                synth_data[fk['ref_table']],
                fk_cols
            )

        dtype_map = {
            col_meta['name']: col_meta['type'].lower()
            for col_meta in schema_tables[tbl]['columns']
        }
        range_map = {
            col: (real_df[col].min(), real_df[col].max())
            for col in real_df.columns
            if pd.api.types.is_numeric_dtype(real_df[col])
        }
        schema_check[tbl] = metrics.compute_schema_adherence(
            synth_df, dtype_map, range_map
        )

    metadata_for_constraints = {
        tbl: {
            'pk':  schema_tables[tbl]['primary_key'],
            'fks': schema_tables[tbl]['foreign_keys']
        }
        for tbl in schema_tables
    }
    constraint_results = metrics.compute_constraint_adherence(
        synth_data, metadata_for_constraints
    )

    full_results = {
        'generation_time_s':    gen_time,
        'statistical_fidelity': stats,
        'distance_metrics':     distances,
        'child_count_ks':       child_ks,
        'schema_adherence':     schema_check,
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