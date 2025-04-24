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


def interpret_results(metrics: dict):
    """
    Print a human‐readable summary of:
      - Univariate statistical fidelity (KS / chi-square + JS)
      - Constraint adherence (PK uniqueness, FK integrity)
    Handles composite FKs (joined by underscores) and empty tables gracefully.
    """
    for table, col_metrics in metrics['statistical'].items():
        print(f"\n=== Table: {table} ===")
        print("— Statistical Fidelity —")
        for col, m in col_metrics.items():
            if m.get('test') == 'ks':
                ks_stat, p = m['ks_stat'], m['p_value']
                verdict = "no significant shift" if p > 0.05 else "shift detected"
                print(f"  • {col}: KS stat={ks_stat:.3f}, p={p:.3f} → {verdict}.")
            else:
                chi2, p_val = m['chi2_stat'], m['chi2_p']
                jsd = m.get('js_divergence')
                print(f"  • {col}: χ²={chi2:.3f}, p={p_val:.3f}, JS={jsd:.4f}.")

        print("— Constraint Adherence —")
        c = metrics['constraints'].get(table, {})
        # Primary key
        pk = c.get('pk_uniqueness')
        if pk is None:
            print("  • PK uniqueness: N/A")
        else:
            print(f"  • PK uniqueness: {pk:.2%} unique rows")

        # Foreign keys
        for key, val in c.items():
            if not key.startswith('fk_'):
                continue
            if val is None:
                print(f"  • FK {cols}: N/A")
            else:
                integrity = "OK" if val > 0.99 else "violations detected"
                print(f"  • FK {key}: {val:.2%} intact → {integrity}")