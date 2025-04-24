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
    metadata: {table: {'pk': [...], 'fks': [{'columns', 'ref_table', 'ref_columns'}]}}
    """
    results = {}
    for table, df in synth.items():
        if len(df) > 0:
            return
        results.setdefault(table, {})
        meta = metadata.get(table, {})
        # PK
        pk_cols = meta.get('pk', [])
        if pk_cols :
            results[table]['pk_uniqueness'] = df.drop_duplicates(pk_cols).shape[0] / len(df)
        else:
            results[table]['pk_uniqueness'] = None

        # FKs
        for fk in meta.get('fks', []):
            child_cols  = fk['columns']
            parent_tbl  = fk['ref_table']
            parent_cols = fk.get('ref_columns') or child_cols

            if isinstance(child_cols, list):
                child_keys = df[child_cols].apply(tuple, axis=1)
                parent_keys = synth[parent_tbl][parent_cols].apply(tuple, axis=1)
                valid = child_keys.isin(parent_keys)
                key_name = '_'.join(child_cols)
            else:
                valid = df[child_cols].isin(synth[parent_tbl][parent_cols])
                key_name = child_cols

            results[table][f'fk_{key_name}'] = valid.mean()

    return results


def interpret_results(metrics: dict):
    """
    Print human-readable interpretations of statistical and constraint results.
    """
    print("=== Statistical Fidelity ===")
    for tbl, cols in metrics['statistical'].items():
        print(f"\nTable: {tbl}")
        for col, m in cols.items():
            if m.get('test') == 'ks':
                p = m['p_value']
                print(f"  • {col}: KS p-value={p:.3f} "
                      + ("PASSED" if p > 0.05 else "SHIFT DETECTED"))
            else:
                js = m['js_divergence']
                print(f"  • {col}: JS divergence={js:.4f} (0=perfect)")

    print("\n=== Constraint Adherence ===")
    for tbl, vals in metrics['constraints'].items():
        print(f"\nTable: {tbl}")
        print(f"  • PK uniqueness: {vals['pk_uniqueness']:.3f}")
        for k, v in vals.items():
            if k.startswith('fk_'):
                print(f"  • {k}: {v:.3f}")