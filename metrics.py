# metrics_researcher.py

"""
Research-grade metrics for evaluating synthetic relational data.

Derived from:
- IRG: Incremental Relational Generator [1]
- SQLSynthGen: Differentially-Private SQL Synthesizer [2]
- Synthetic Data Generation for Enterprise DBMS [3]

Metrics:
  4.2 Statistical fidelity (KS, chi-square, child-count KS)
  4.3 Distance-based (TV, KL, JS, Wasserstein, MMD)
  4.4 Schema & constraint adherence (dtype, range, PK, FK)
  4.1 Performance & domain checks (generation time, dialect coverage)
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency, wasserstein_distance, entropy
from sklearn.metrics.pairwise import rbf_kernel


def compute_statistical_fidelity(real: pd.Series, synth: pd.Series):
    """
    - Numeric → two-sample KS test
    - Categorical → chi-square (Laplace smoothing) + JS divergence
    """
    real = real.dropna()
    synth = synth.dropna()

    # KS on numeric
    real_num = pd.to_numeric(real, errors='coerce').dropna()
    synth_num = pd.to_numeric(synth, errors='coerce').dropna()
    if len(real_num) and len(synth_num):
        stat, p = ks_2samp(real_num, synth_num)
        return {'test': 'ks', 'ks_stat': stat, 'p_value': p}

    # Chi-square + JS on categorical
    r_counts = real.astype(str).value_counts()
    s_counts = synth.astype(str).value_counts()
    idx = r_counts.index.union(s_counts.index)
    r = r_counts.reindex(idx, fill_value=0).values + 1
    s = s_counts.reindex(idx, fill_value=0).values + 1

    chi2, chi2_p, _, _ = chi2_contingency(np.stack([r, s], axis=1))
    p_norm = r / r.sum()
    q_norm = s / s.sum()
    m = 0.5 * (p_norm + q_norm)
    jsd = 0.5 * (entropy(p_norm, m) + entropy(q_norm, m))

    return {
        'test': 'chi2_jsd',
        'chi2_stat': chi2,
        'chi2_p': chi2_p,
        'js_divergence': jsd
    }


def compute_child_count_ks(real_child: pd.DataFrame,
                           synth_child: pd.DataFrame,
                           fk_cols):
    """
    KS on distribution of # child rows per parent key.
    fk_cols may be a single column or list of columns.
    """
    if isinstance(fk_cols, list):
        rc = real_child.groupby(fk_cols).size()
        sc = synth_child.groupby(fk_cols).size()
    else:
        rc = real_child.groupby(fk_cols).size()
        sc = synth_child.groupby(fk_cols).size()
    stat, p = ks_2samp(rc, sc)
    return {'child_ks_stat': stat, 'child_ks_p': p}


def compute_distance_metrics(real: pd.Series, synth: pd.Series):
    """
    Compute:
      - Total Variation (TV)
      - KL divergence
      - JS divergence
      - Wasserstein (numeric only)
      - MMD (numeric only)
    """
    real_counts = real.fillna('NA').astype(str).value_counts(normalize=True)
    synth_counts = synth.fillna('NA').astype(str).value_counts(normalize=True)
    idx = real_counts.index.union(synth_counts.index)
    p = real_counts.reindex(idx, fill_value=0).values
    q = synth_counts.reindex(idx, fill_value=0).values

    tv = 0.5 * np.sum(np.abs(p - q))
    kl = entropy(p + 1e-12, q + 1e-12)
    m = 0.5 * (p + q)
    js = 0.5 * (entropy(p + 1e-12, m) + entropy(q + 1e-12, m))

    wass = None
    mmd = None

    if pd.api.types.is_numeric_dtype(real) and pd.api.types.is_numeric_dtype(synth):
        real_num = pd.to_numeric(real, errors='coerce').dropna().astype(float)
        synth_num = pd.to_numeric(synth, errors='coerce').dropna().astype(float)
        if len(real_num) and len(synth_num):
            wass = wasserstein_distance(real_num, synth_num)
            X = real_num.values.reshape(-1, 1)
            Y = synth_num.values.reshape(-1, 1)
            Kxx = rbf_kernel(X, X)
            Kyy = rbf_kernel(Y, Y)
            Kxy = rbf_kernel(X, Y)
            mmd = (Kxx.sum() / (len(X)**2)
                   + Kyy.sum() / (len(Y)**2)
                   - 2 * Kxy.sum() / (len(X) * len(Y)))

    return {'tv': tv, 'kl': kl, 'js': js, 'wasserstein': wass, 'mmd': mmd}


def compute_schema_adherence(df: pd.DataFrame,
                             dtype_map: dict,
                             range_map: dict):
    """
    Check data-type validity and numeric range adherence.
    """
    dtype_results = {}
    for col, dt in dtype_map.items():
        series = df[col]
        if dt.startswith('int'):
            nums = pd.to_numeric(series, errors='coerce').dropna().astype(float)
            valid = nums.apply(lambda x: x.is_integer())
        elif dt in ('float', 'real'):
            valid = pd.to_numeric(series, errors='coerce').notna()
        elif dt in ('date', 'timestamp'):
            valid = pd.to_datetime(series, errors='coerce').notna()
        else:
            valid = series.notna()
        dtype_results[col] = valid.mean()

    range_results = {}
    for col, (lo, hi) in range_map.items():
        nums = pd.to_numeric(df[col], errors='coerce').dropna().astype(float)
        if lo is not None and hi is not None:
            in_range = nums.between(lo, hi)
        else:
            in_range = pd.Series(True, index=nums.index)
        range_results[col] = in_range.mean()

    return {'dtype_validity': dtype_results,
            'range_adherence': range_results}

def compute_constraint_adherence(synth: dict, metadata: dict):
    """
    Check PK uniqueness and FK integrity.
    metadata: {
      table: {
        'pk': [...],
        'fks': [
          {'columns': [...], 'ref_table': '...', 'ref_columns': [...]}, ...
        ]
      }
    }
    """
    results = {}
    for table, df in synth.items():
        results.setdefault(table, {})

        # If the synthetic table is empty, skip detailed checks
        if df.empty:
            results[table]['pk_uniqueness'] = None
            continue

        meta = metadata.get(table, {})

        # --- Primary Key Uniqueness ---
        pk_cols = meta.get('pk', [])
        if pk_cols:
            unique_count = df.drop_duplicates(subset=pk_cols).shape[0]
            results[table]['pk_uniqueness'] = unique_count / len(df)
        else:
            results[table]['pk_uniqueness'] = None

        # --- Foreign Key Integrity ---
        for fk in meta.get('fks', []):
            child_cols  = fk.get('columns', [])
            parent_tbl  = fk.get('ref_table')
            parent_cols = fk.get('ref_columns') or child_cols
            parent_df   = synth.get(parent_tbl)

            # Skip if no valid FK definition or missing parent table
            if not child_cols or parent_df is None:
                continue

            # Handle composite vs. single-column FKs
            if isinstance(child_cols, (list, tuple)):
                # build tuple keys
                child_keys  = df[child_cols].apply(lambda row: tuple(row), axis=1)
                parent_keys = parent_df[parent_cols].apply(lambda row: tuple(row), axis=1)
                key_name    = "_".join(child_cols)
            else:
                child_keys  = df[child_cols]
                parent_keys = parent_df[parent_cols]
                key_name    = child_cols

            valid_fraction = child_keys.isin(parent_keys).mean()
            results[table][f"fk_{key_name}"] = valid_fraction

    return results


import pandas as pd

def interpret_full_results(full_results: dict):
    """
    Build and display summary DataFrames for all experiment metrics,
    then print human‐readable interpretation.
    """
    # 1) Generation time
    gen_time = full_results.get('generation_time_s')
    print(f"\nData generation time: {gen_time:.2f} seconds\n")

    # 2) Statistical fidelity
    stat_rows = []
    for table, cols in full_results['statistical_fidelity'].items():
        for col, m in cols.items():
            if m.get('test') == 'ks':
                stat_rows.append({
                    'Table': table, 'Column': col, 'Test': 'KS',
                    'Statistic': m['ks_stat'], 'P-value': m['p_value'],
                    'JS divergence': None
                })
            else:
                stat_rows.append({
                    'Table': table, 'Column': col, 'Test': 'Chi²+JS',
                    'Statistic': m['chi2_stat'], 'P-value': m['chi2_p'],
                    'JS divergence': m['js_divergence']
                })
    stats_df = pd.DataFrame(stat_rows)
    print("--- Statistical Fidelity ---")
    print(stats_df.to_string(index=False))

    # 3) Distance metrics
    dist_rows = []
    for table, cols in full_results['distance_metrics'].items():
        for col, d in cols.items():
            dist_rows.append({
                'Table': table, 'Column': col,
                'TV': d['tv'], 'KL': d['kl'],
                'JS': d['js'],
                'Wasserstein': d['wasserstein'],
                'MMD': d['mmd']
            })
    dist_df = pd.DataFrame(dist_rows)
    print("\n--- Distance Metrics ---")
    print(dist_df.to_string(index=False))

    # 4) Child-count KS
    child_rows = []
    for table, fks in full_results['child_count_ks'].items():
        for fk_key, m in fks.items():
            child_rows.append({
                'Table': table, 'FK': fk_key,
                'KS stat': m['child_ks_stat'],
                'P-value': m['child_ks_p']
            })
    child_df = pd.DataFrame(child_rows)
    print("\n--- Child-Count KS for FKs ---")
    print(child_df.to_string(index=False))

    # 5) Schema & range adherence
    schema_rows = []
    for table, sc in full_results['schema_adherence'].items():
        for col, v in sc['dtype_validity'].items():
            schema_rows.append({
                'Table': table, 'Column': col,
                'Check': 'Type validity', 'Value': v
            })
        for col, v in sc['range_adherence'].items():
            schema_rows.append({
                'Table': table, 'Column': col,
                'Check': 'Range adherence', 'Value': v
            })
    schema_df = pd.DataFrame(schema_rows)
    print("\n--- Schema & Range Adherence ---")
    print(schema_df.to_string(index=False))

    # 6) Constraint adherence
    cons_rows = []
    for table, c in full_results['constraint_adherence'].items():
        pk = c.get('pk_uniqueness')
        cons_rows.append({
            'Table': table, 'Constraint': 'PK uniqueness',
            'Value': pk
        })
        for key, v in c.items():
            if key.startswith('fk_'):
                cons_rows.append({
                    'Table': table, 'Constraint': key,
                    'Value': v
                })
    cons_df = pd.DataFrame(cons_rows)
    print("\n--- Constraint Adherence ---")
    print(cons_df.to_string(index=False))

    # 7) Interpretation
    print("\nInterpretation:")
    print(" • KS P-value > 0.05 → no significant numeric shift.")
    print(" • JS divergence near 0 → categorical distributions match closely.")
    print(" • TV, KL small → high overall fidelity; Wasserstein & MMD closer to 0 → better.")
    print(" • Child-count KS P-value > 0.05 → FK relationship counts preserved.")
    print(" • Type validity = fraction of values matching declared SQL type.")
    print(" • Range adherence = fraction within real-data min/max bounds.")
    print(" • PK uniqueness = fraction of unique primary keys (1.00 = perfect).")
    print(" • FK integrity = fraction of child rows pointing to existing parent keys (1.00 = perfect).")