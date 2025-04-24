# metrics_general.py

"""
General-purpose metrics for evaluating synthetic relational data against real data.

Includes:
- Univariate distribution tests (numeric, categorical, datetime, boolean)
- Bivariate relationship test (child‐count KS)
- Distance‐based metrics: TV, KL, JS, Wasserstein, MMD
- Schema & constraint adherence: data type validity, range adherence, PK uniqueness, FK integrity

Dependencies:
    pip install pandas scipy numpy scikit-learn
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency, entropy, wasserstein_distance
from sklearn.metrics.pairwise import rbf_kernel


# --- 4.2 Statistical Fidelity ---

def univariate_test(real: pd.Series, synth: pd.Series):
    """
    Unified univariate distribution test:
    - Numeric → KS test
    - Categorical or boolean → Chi-square + JS divergence
    - Datetime      → KS on ordinal
    Returns a dict with test name and statistics.
    """
    # drop nulls
    real_nonnull = real.dropna()
    synth_nonnull = synth.dropna()

    # Numeric
    if pd.api.types.is_numeric_dtype(real_nonnull):
        r = real_nonnull.astype(float)
        s = synth_nonnull.astype(float)
        stat, p = ks_2samp(r, s)
        return {'test': 'ks', 'stat': stat, 'p_value': p}

    # Datetime
    if pd.api.types.is_datetime64_any_dtype(real_nonnull):
        # convert to integer timestamps
        r = real_nonnull.astype(np.int64)
        s = synth_nonnull.astype(np.int64)
        stat, p = ks_2samp(r, s)
        return {'test': 'ks_datetime', 'stat': stat, 'p_value': p}

    # Boolean
    if pd.api.types.is_bool_dtype(real_nonnull):
        # treat as categorical
        real_nonnull = real_nonnull.astype(str)
        synth_nonnull = synth_nonnull.astype(str)
        # fall through to categorical

    # Categorical (objects, categories, bool after cast)
    real_counts = real_nonnull.astype(str).value_counts(normalize=True)
    synth_counts = synth_nonnull.astype(str).value_counts(normalize=True)
    idx = real_counts.index.union(synth_counts.index)
    p = real_counts.reindex(idx, fill_value=0).values
    q = synth_counts.reindex(idx, fill_value=0).values

    # Chi-square test
    contingency = np.stack([
        real_counts.reindex(idx, fill_value=0).values,
        synth_counts.reindex(idx, fill_value=0).values
    ], axis=1)
    chi2, chi2_p, _, _ = chi2_contingency(contingency)

    # Jensen-Shannon divergence
    eps = 1e-12
    p_safe = p + eps
    q_safe = q + eps
    m = 0.5 * (p_safe + q_safe)
    jsd = 0.5 * (entropy(p_safe, m) + entropy(q_safe, m))

    return {
        'test': 'chi2_jsd',
        'chi2_stat': chi2,
        'chi2_p': chi2_p,
        'js_divergence': jsd
    }


def child_count_ks(real_child: pd.DataFrame, synth_child: pd.DataFrame,
                   fk: str):
    """
    KS test on distribution of number of child rows per parent.
    fk: foreign-key column name in the child table.
    """
    real_counts = real_child.groupby(fk).size()
    synth_counts = synth_child.groupby(fk).size()
    stat, p = ks_2samp(real_counts, synth_counts)
    return {'child_ks_stat': stat, 'child_ks_p': p}


# --- 4.3 Distance-Based Metrics ---

def total_variation(p: np.ndarray, q: np.ndarray):
    """Total Variation Distance between two discrete distributions."""
    return 0.5 * np.sum(np.abs(p - q))


def kullback_leibler(p: np.ndarray, q: np.ndarray, eps=1e-12):
    """Kullback–Leibler divergence D(P‖Q)."""
    p = p + eps
    q = q + eps
    return entropy(p, q)


def jensen_shannon(p: np.ndarray, q: np.ndarray, eps=1e-12):
    """Jensen–Shannon divergence between distributions."""
    p = p + eps
    q = q + eps
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))


def wasserstein(real: pd.Series, synth: pd.Series):
    """Wasserstein (Earth Mover’s) distance for numeric variables."""
    return wasserstein_distance(real.dropna(), synth.dropna())


def maximum_mean_discrepancy(real: np.ndarray, synth: np.ndarray, gamma=1.0):
    """
    Compute MMD with an RBF kernel:
    MMD^2 = E[K(x,x')] + E[K(y,y')] - 2 E[K(x,y')]
    """
    X = real.reshape(-1, 1)
    Y = synth.reshape(-1, 1)
    Kxx = rbf_kernel(X, X, gamma=gamma)
    Kyy = rbf_kernel(Y, Y, gamma=gamma)
    Kxy = rbf_kernel(X, Y, gamma=gamma)
    m, n = X.shape[0], Y.shape[0]
    return Kxx.sum()/(m*m) + Kyy.sum()/(n*n) - 2*Kxy.sum()/(m*n)


# --- 4.4 Schema & Constraint Adherence ---

def dtype_validity(df: pd.DataFrame, schema: dict):
    """
    Check that each column’s values conform to the expected dtype in schema.
    schema: {col: dtype}, dtype strings like 'int', 'float', 'str', 'date', 'bool'.
    Returns fraction valid per column.
    """
    results = {}
    for col, dt in schema.items():
        series = df[col]
        if dt in ('int', 'smallint'):
            valid = series.dropna().apply(lambda x: isinstance(x, (int, np.integer)))
        elif dt in ('float', 'double', 'real'):
            valid = series.dropna().apply(lambda x: isinstance(x, (float, int, np.floating)))
        elif dt in ('str', 'varchar', 'text', 'char'):
            valid = series.dropna().apply(lambda x: isinstance(x, str))
        elif dt in ('date', 'timestamp'):
            valid = pd.to_datetime(series, errors='coerce').notna()
        elif dt == 'bool':
            valid = series.dropna().apply(lambda x: isinstance(x, (bool, np.bool_)))
        else:
            # fallback: non-null
            valid = series.notna()
        results[col] = valid.mean()
    return results


def range_adherence(df: pd.DataFrame, bounds: dict):
    """
    Check numeric/date columns stay within specified bounds.
    bounds: {col: (min_value, max_value)}.
    Returns fraction in-range per column.
    """
    results = {}
    for col, (min_v, max_v) in bounds.items():
        series = df[col].dropna()
        if pd.api.types.is_numeric_dtype(series):
            s = series.astype(float)
        else:
            s = pd.to_datetime(series, errors='coerce')
        in_range = s.between(min_v, max_v)
        results[col] = in_range.mean()
    return results


def pk_uniqueness(df: pd.DataFrame, pk_cols):
    """
    Fraction of unique primary key combinations.
    pk_cols: single column name or list of columns.
    """
    total = len(df)
    unique = df.drop_duplicates(subset=pk_cols).shape[0]
    return unique / total if total else 0.0


def fk_integrity(df_child: pd.DataFrame, df_parent: pd.DataFrame,
                 fk_col: str, parent_pk: str):
    """
    Fraction of child rows whose fk_col exists in parent[parent_pk].
    """
    valid = df_child[fk_col].isin(df_parent[parent_pk])
    return valid.mean()