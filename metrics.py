import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency, wasserstein_distance, entropy
from sklearn.metrics.pairwise import rbf_kernel


def load_csv_dir(directory: str) -> dict:
    """
    Load all CSV files in a directory into a dict of DataFrames keyed by filename (without extension).
    """
    return {
        os.path.splitext(os.path.basename(path))[0]: pd.read_csv(path)
        for path in glob.glob(os.path.join(directory, '*.csv'))
    }


def ks_and_chi2(real: pd.Series, synth: pd.Series, table: str, column: str) -> pd.DataFrame:
    """
    Compute two-sample KS for numeric data and chi-square + JS divergence for categorical data.
    Returns a long DataFrame with columns: table, column, metric, value.
    """
    real_clean = real.dropna()
    synth_clean = synth.dropna()
    rows = []

    # KS for numeric
    real_num = pd.to_numeric(real_clean, errors='coerce').dropna()
    synth_num = pd.to_numeric(synth_clean, errors='coerce').dropna()
    if len(real_num) and len(synth_num):
        stat, p = ks_2samp(real_num, synth_num)
        rows.append((table, column, 'ks_stat', stat))
        rows.append((table, column, 'p_value', p))

    # Chi-square + JS for categorical
    r_counts = real_clean.astype(str).value_counts()
    s_counts = synth_clean.astype(str).value_counts()
    idx = r_counts.index.union(s_counts.index)
    if not idx.empty:
        r = (r_counts.reindex(idx, fill_value=0) + 1).values
        s = (s_counts.reindex(idx, fill_value=0) + 1).values
        chi2, chi2_p, _, _ = chi2_contingency(np.stack([r, s], axis=1))
        rows.append((table, column, 'chi2_stat', chi2))
        rows.append((table, column, 'chi2_p', chi2_p))
        p_norm = r / r.sum()
        q_norm = s / s.sum()
        m = 0.5 * (p_norm + q_norm)
        jsd = 0.5 * (entropy(p_norm, m) + entropy(q_norm, m))
        rows.append((table, column, 'js_divergence', jsd))

    return pd.DataFrame(rows, columns=['table', 'column', 'metric', 'value'])


def distance_metrics_df(real: pd.Series, synth: pd.Series, table: str, column: str) -> pd.DataFrame:
    """
    Compute TV, KL, JS divergence for any column; plus Wasserstein & MMD if numeric.
    Returns a long DataFrame with columns: table, column, metric, value.
    """
    real_str = real.fillna('NA').astype(str)
    synth_str = synth.fillna('NA').astype(str)
    real_counts = real_str.value_counts(normalize=True)
    synth_counts = synth_str.value_counts(normalize=True)
    idx = real_counts.index.union(synth_counts.index)
    p = real_counts.reindex(idx, fill_value=0).values
    q = synth_counts.reindex(idx, fill_value=0).values

    rows = []
    tv = 0.5 * np.sum(np.abs(p - q))
    kl = entropy(p + 1e-12, q + 1e-12)
    m = 0.5 * (p + q)
    js = 0.5 * (entropy(p + 1e-12, m) + entropy(q + 1e-12, m))
    rows.extend([
        (table, column, 'tv', tv),
        (table, column, 'kl', kl),
        (table, column, 'js', js),
    ])

    # Numeric-specific metrics
    if pd.api.types.is_numeric_dtype(real) and pd.api.types.is_numeric_dtype(synth):
        rn = pd.to_numeric(real, errors='coerce').dropna().astype(float)
        sn = pd.to_numeric(synth, errors='coerce').dropna().astype(float)
        if len(rn) and len(sn):
            wass = wasserstein_distance(rn, sn)
            X, Y = rn.values.reshape(-1, 1), sn.values.reshape(-1, 1)
            Kxx, Kyy, Kxy = rbf_kernel(X, X), rbf_kernel(Y, Y), rbf_kernel(X, Y)
            mmd = (Kxx.sum() / (len(X)**2)
                   + Kyy.sum() / (len(Y)**2)
                   - 2 * Kxy.sum() / (len(X)*len(Y)))
            rows.append((table, column, 'wasserstein', wass))
            rows.append((table, column, 'mmd', mmd))

    return pd.DataFrame(rows, columns=['table', 'column', 'metric', 'value'])


def child_count_ks_df(real_data: dict, synth_data: dict, schema: dict) -> pd.DataFrame:
    """
    For each foreign-key in schema, compares child-count distributions via KS.
    Returns table, fk, metric, value.
    """
    rows = []
    for table, meta in schema.items():
        for fk in meta.get('foreign_keys', []):
            fk_cols = fk['columns']
            parent = fk['ref_table']
            if parent not in real_data or parent not in synth_data:
                continue
            rc = real_data[parent].groupby(fk_cols).size()
            sc = synth_data[parent].groupby(fk_cols).size()
            stat, p = ks_2samp(rc, sc)
            fk_name = '_'.join(fk_cols) if isinstance(fk_cols, (list, tuple)) else fk_cols
            rows.append((table, fk_name, 'child_ks_stat', stat))
            rows.append((table, fk_name, 'child_ks_p', p))
    return pd.DataFrame(rows, columns=['table', 'fk', 'metric', 'value'])


def schema_adherence_df(synth_data: dict, schema: dict, real_data: dict) -> pd.DataFrame:
    """
    Compute dtype validity and range adherence per column.
    Returns table, column, check, metric, value.
    """
    rows = []
    for table, df in synth_data.items():
        meta = schema.get(table, {})
        # dtype_map from schema (fallback to no columns)
        dtype_map = {c['name']: c['type'].lower() for c in meta.get('columns', [])}
        # get real_df if available
        real_df = real_data.get(table)
        if real_df is not None:
            range_map = {
                col: (real_df[col].min(), real_df[col].max())
                for col in real_df.columns
                if pd.api.types.is_numeric_dtype(real_df[col])
            }
        else:
            range_map = {}

        # dtype validity
        for col, dt in dtype_map.items():
            series = df[col]
            if dt.startswith('int'):
                valid = pd.to_numeric(series, errors='coerce').dropna().apply(lambda x: float(x).is_integer()).mean()
            elif dt in ('float', 'real'):
                valid = pd.to_numeric(series, errors='coerce').notna().mean()
            elif dt in ('date', 'timestamp'):
                valid = pd.to_datetime(series, errors='coerce').notna().mean()
            else:
                valid = series.notna().mean()
            rows.append((table, col, 'dtype', 'validity', valid))

        # range adherence
        for col, (lo, hi) in range_map.items():
            nums = pd.to_numeric(df[col], errors='coerce').dropna().astype(float)
            in_range = nums.between(lo, hi).mean() if lo is not None and hi is not None else np.nan
            rows.append((table, col, 'range', 'adherence', in_range))

    return pd.DataFrame(rows, columns=['table', 'column', 'check', 'metric', 'value'])


def constraint_metrics_df(synth_data: dict, schema: dict) -> pd.DataFrame:
    """
    Compute PK uniqueness and FK validity for each table.
    Returns table, constraint, metric, value.
    """
    rows = []
    for table, df in synth_data.items():
        meta = schema.get(table, {})
        pk = meta.get('primary_key', [])
        if pk and len(df):
            frac = df.drop_duplicates(subset=pk).shape[0] / len(df)
            rows.append((table, 'pk_uniqueness', 'value', frac))
        for fk in meta.get('foreign_keys', []):
            child_cols = fk['columns']
            parent = fk['ref_table']
            if parent not in synth_data:
                continue
            child_keys = df[child_cols].apply(tuple, axis=1)
            parent_keys = synth_data[parent][fk['ref_columns']].apply(tuple, axis=1)
            frac = child_keys.isin(parent_keys).mean()
            fk_name = '_'.join(child_cols) if isinstance(child_cols, (list, tuple)) else child_cols
            rows.append((table, f'fk_{fk_name}', 'value', frac))
    return pd.DataFrame(rows, columns=['table', 'constraint', 'metric', 'value'])


# Interpretation rules by metric
interpret_rules = {
    'ks_stat':        lambda v: 'no shift' if v > 0.05 else 'shift detected',
    'p_value':        lambda v: 'match' if v > 0.05 else 'difference',
    'chi2_p':         lambda v: 'match' if v > 0.05 else 'difference',
    'js_divergence':  lambda v: 'high fidelity' if v < 0.1 else 'divergence',
    'tv':             lambda v: 'high fidelity' if v < 0.1 else 'noticeable divergence',
    'kl':             lambda v: 'high fidelity' if v < 0.1 else 'noticeable divergence',
    'js':             lambda v: 'high fidelity' if v < 0.1 else 'noticeable divergence',
    'wasserstein':    lambda v: 'small distance' if v < 1 else 'larger distance',
    'mmd':            lambda v: 'small MMD' if v < 1e-3 else 'larger MMD',
    'child_ks_stat':  lambda v: 'counts match' if v < 0.05 else 'counts differ',
    'child_ks_p':     lambda v: 'counts match' if v > 0.05 else 'counts differ',
    'validity':       lambda v: 'OK' if v > 0.99 else 'violations',
    'adherence':      lambda v: 'OK' if v > 0.99 else 'violations',
    'value':          lambda v: 'OK' if v > 0.99 else 'violations',
}


def interpret(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add an 'interpretation' column based on metric and value.
    """
    def apply_rule(row):
        metric = row['metric']
        value = row['value']
        rule = interpret_rules.get(metric)
        return rule(value) if rule is not None else None

    df['interpretation'] = df.apply(apply_rule, axis=1)
    return df


def apply_per_column(real_data: dict, synth_data: dict, func, **extra) -> pd.DataFrame:
    """
    Apply a function over each table and each column of real vs synth data.
    """
    frames = []
    for table, real_df in real_data.items():
        synth_df = synth_data.get(table)
        if synth_df is None:
            continue
        for col in real_df.columns:
            frames.append(func(real_df[col], synth_df[col], table=table, column=col, **extra))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


class SyntheticDataEvaluator:
    def __init__(self, real_dir: str, synth_dir: str, schema: dict):
        self.real_data = load_csv_dir(real_dir)
        self.synth_data = load_csv_dir(synth_dir)
        self.schema = schema

    def evaluate(self) -> dict:
        stats = apply_per_column(self.real_data, self.synth_data, ks_and_chi2)
        dist  = apply_per_column(self.real_data, self.synth_data, distance_metrics_df)
        child = child_count_ks_df(self.real_data, self.synth_data, self.schema)
        schema_chk = schema_adherence_df(self.synth_data, self.schema, self.real_data)
        cons  = constraint_metrics_df(self.synth_data, self.schema)

        return {
            'statistical_fidelity': interpret(stats),
            'distance_metrics':     interpret(dist),
            'child_count_ks':       interpret(child),
            'schema_adherence':     interpret(schema_chk),
            'constraint_adherence': interpret(cons),
        }

    def save_to_csv(self, results: dict, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        for name, df in results.items():
            df.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)


def evaluate_from_csv_dirs(real_dir: str,
                           synth_dir: str,
                           schema_tables: dict,
                           output_dir: str) -> dict:
    """
    Convenience wrapper: load data, run all metrics, save CSVs, and return DataFrames.
    """
    evaluator = SyntheticDataEvaluator(real_dir, synth_dir, schema_tables)
    results = evaluator.evaluate()
    evaluator.save_to_csv(results, output_dir)
    return results
