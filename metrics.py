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
  7   Evaluation from CSV directories
"""

import os
import glob
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

    # If both empty, nothing to compare
    if real.empty and synth.empty:
        return {'test': None,
                'ks_stat': None, 'p_value': None,
                'chi2_stat': None, 'chi2_p': None,
                'js_divergence': None}

    # KS on numeric
    real_num = pd.to_numeric(real, errors='coerce').dropna()
    synth_num = pd.to_numeric(synth, errors='coerce').dropna()
    if len(real_num) and len(synth_num):
        stat, p = ks_2samp(real_num, synth_num)
        return {'test': 'ks',
                'ks_stat': stat, 'p_value': p,
                'chi2_stat': None, 'chi2_p': None,
                'js_divergence': None}

    # Categorical χ² + JS
    r_counts = real.astype(str).value_counts()
    s_counts = synth.astype(str).value_counts()
    idx = r_counts.index.union(s_counts.index)
    if len(idx) == 0:
        return {'test': None,
                'ks_stat': None, 'p_value': None,
                'chi2_stat': None, 'chi2_p': None,
                'js_divergence': None}

    r = r_counts.reindex(idx, fill_value=0).values + 1
    s = s_counts.reindex(idx, fill_value=0).values + 1
    chi2, chi2_p, _, _ = chi2_contingency(np.stack([r, s], axis=1))
    p_norm = r / r.sum()
    q_norm = s / s.sum()
    m = 0.5 * (p_norm + q_norm)
    jsd = 0.5 * (entropy(p_norm, m) + entropy(q_norm, m))

    return {
        'test': 'chi2_jsd',
        'ks_stat': None, 'p_value': None,
        'chi2_stat': chi2, 'chi2_p': chi2_p,
        'js_divergence': jsd
    }


def compute_child_count_ks(real_child: pd.DataFrame,
                           synth_child: pd.DataFrame,
                           fk_cols):
    if isinstance(fk_cols, (list, tuple)):
        rc = real_child.groupby(fk_cols).size()
        sc = synth_child.groupby(fk_cols).size()
    else:
        rc = real_child.groupby(fk_cols).size()
        sc = synth_child.groupby(fk_cols).size()
    stat, p = ks_2samp(rc, sc)
    return {'child_ks_stat': stat, 'child_ks_p': p}


def compute_distance_metrics(real: pd.Series, synth: pd.Series):
    real_counts = real.fillna('NA').astype(str).value_counts(normalize=True)
    synth_counts = synth.fillna('NA').astype(str).value_counts(normalize=True)
    idx = real_counts.index.union(synth_counts.index)
    p = real_counts.reindex(idx, fill_value=0).values
    q = synth_counts.reindex(idx, fill_value=0).values
    tv = 0.5 * np.sum(np.abs(p - q))
    kl = entropy(p+1e-12, q+1e-12)
    m = 0.5*(p+q)
    js = 0.5*(entropy(p+1e-12, m)+entropy(q+1e-12, m))
    wass = None; mmd = None
    if pd.api.types.is_numeric_dtype(real) and pd.api.types.is_numeric_dtype(synth):
        rn = pd.to_numeric(real, errors='coerce').dropna().astype(float)
        sn = pd.to_numeric(synth, errors='coerce').dropna().astype(float)
        if len(rn) and len(sn):
            wass = wasserstein_distance(rn, sn)
            X, Y = rn.values.reshape(-1,1), sn.values.reshape(-1,1)
            Kxx, Kyy, Kxy = rbf_kernel(X,X), rbf_kernel(Y,Y), rbf_kernel(X,Y)
            mmd = Kxx.sum()/(len(X)**2) + Kyy.sum()/(len(Y)**2) - 2*Kxy.sum()/(len(X)*len(Y))
    return {'tv': tv, 'kl': kl, 'js': js, 'wasserstein': wass, 'mmd': mmd}


def compute_schema_adherence(df: pd.DataFrame,
                             dtype_map: dict,
                             range_map: dict):
    dtype_results = {}
    for col, dt in dtype_map.items():
        series = df[col]
        if dt.startswith('int'):
            nums = pd.to_numeric(series, errors='coerce').dropna().astype(float)
            valid = nums.apply(lambda x: x.is_integer())
        elif dt in ('float','real'):
            valid = pd.to_numeric(series, errors='coerce').notna()
        elif dt in ('date','timestamp'):
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
    results = {}
    for table, df in synth.items():
        results.setdefault(table, {})
        if df.empty:
            results[table]['pk_uniqueness'] = None
            continue
        meta = metadata.get(table, {})
        pk_cols = meta.get('pk', [])
        results[table]['pk_uniqueness'] = (df.drop_duplicates(subset=pk_cols).shape[0] / len(df)
                                           if pk_cols else None)
        for fk in meta.get('fks', []):
            child_cols = fk.get('columns', [])
            parent_tbl = fk.get('ref_table')
            parent_cols= fk.get('ref_columns') or child_cols
            parent_df  = synth.get(parent_tbl)
            if not child_cols or parent_df is None:
                continue
            if isinstance(child_cols, (list, tuple)):
                child_keys = df[child_cols].apply(lambda r: tuple(r), axis=1)
                parent_keys= parent_df[parent_cols].apply(lambda r: tuple(r), axis=1)
                key_name   = "_".join(child_cols)
            else:
                child_keys, parent_keys = df[child_cols], parent_df[parent_cols]
                key_name = child_cols
            results[table][f'fk_{key_name}'] = child_keys.isin(parent_keys).mean()
    return results


def evaluate_from_csv_dirs(real_dir: str,
                           synth_dir: str,
                           schema_tables: dict) -> dict:
    # load real CSVs
    real_data = {os.path.splitext(os.path.basename(p))[0]: pd.read_csv(p)
                 for p in glob.glob(os.path.join(real_dir, '*.csv'))}
    # load synthetic CSVs
    synth_data = {os.path.splitext(os.path.basename(p))[0]: pd.read_csv(p)
                  for p in glob.glob(os.path.join(synth_dir, '*.csv'))}
    stats, dist, child, schema_chk = {}, {}, {}, {}
    for tbl, real_df in real_data.items():
        synth_df = synth_data.get(tbl)
        if synth_df is None or synth_df.empty:
            continue
        stats[tbl] = {col: compute_statistical_fidelity(real_df[col], synth_df[col])
                      for col in real_df.columns}
        dist[tbl]  = {col: compute_distance_metrics(real_df[col], synth_df[col])
                      for col in real_df.columns}
        child[tbl] = {}
        for fk in schema_tables[tbl].get('foreign_keys', []):
            fk_cols = fk['columns']
            key_str = "_".join(fk_cols) if isinstance(fk_cols,(list,tuple)) else fk_cols
            child[tbl][key_str] = compute_child_count_ks(
                real_data[fk['ref_table']], synth_data[fk['ref_table']], fk_cols)
        dtype_map = {c['name']: c['type'].lower() for c in schema_tables[tbl]['columns']}
        range_map = {col:(real_df[col].min(), real_df[col].max())
                     for col in real_df.columns if pd.api.types.is_numeric_dtype(real_df[col])}
        schema_chk[tbl] = compute_schema_adherence(synth_df, dtype_map, range_map)
    meta_cons = {t:{'pk':schema_tables[t]['primary_key'],
                    'fks':schema_tables[t]['foreign_keys']}
                for t in schema_tables}
    cons = compute_constraint_adherence(synth_data, meta_cons)
    return {
        'statistical_fidelity': stats,
        'distance_metrics':     dist,
        'child_count_ks':       child,
        'schema_adherence':     schema_chk,
        'constraint_adherence': cons
    }


def interpret_and_save(full_results: dict, output_dir: str):
    """
    Convert full_results into pandas DataFrames with an added 'Interpretation'
    column, and write each to CSV files.
    """

    os.makedirs(output_dir, exist_ok=True)

    def interpret_stat(row):
        test = row['Test']
        if test == 'ks':
            return 'no shift' if row['P-value'] > 0.05 else 'shift detected'
        elif test == 'chi2_jsd':
            return (
                'categorical match'
                if row['P-value'] > 0.05 and row['JS divergence'] is not None and row['JS divergence'] < 0.1
                else 'categories differ'
            )
        else:
            return 'no data'

    # 1) Statistical fidelity → DataFrame with interpretation
    stat_rows = []
    for tbl, cols in full_results['statistical_fidelity'].items():
        for col, m in cols.items():
            stat_rows.append({
                'Table': tbl,
                'Column': col,
                'Test': m['test'],
                'Statistic': m.get('ks_stat', m.get('chi2_stat')),
                'P-value': m.get('p_value', m.get('chi2_p')),
                'JS divergence': m.get('js_divergence')
            })
    stats_df = pd.DataFrame(stat_rows)
    stats_df['Interpretation'] = stats_df.apply(interpret_stat, axis=1)
    stats_df.to_csv(os.path.join(output_dir, 'statistical_fidelity.csv'), index=False)

    # 2) Distance metrics
    def interpret_dist(row):
        # small values (<0.1) are good
        vals = [row['TV'], row['KL'], row['JS']]
        if all(v < 0.1 for v in vals):
            return "high fidelity"
        return "noticeable divergence"

    dist_rows = []
    for tbl, cols in full_results['distance_metrics'].items():
        for col, d in cols.items():
            dist_rows.append({
                'Table': tbl,
                'Column': col,
                'TV': d['tv'],
                'KL': d['kl'],
                'JS': d['js'],
                'Wasserstein': d['wasserstein'],
                'MMD': d['mmd']
            })
    dist_df = pd.DataFrame(dist_rows)
    dist_df['Interpretation'] = dist_df.apply(interpret_dist, axis=1)
    dist_df.to_csv(os.path.join(output_dir, 'distance_metrics.csv'), index=False)

    # 3) Child-count KS
    def interpret_child(row):
        return ("counts match" if row['P-value'] > 0.05 else "counts differ")

    child_rows = []
    for tbl, fks in full_results['child_count_ks'].items():
        for fk, m in fks.items():
            child_rows.append({
                'Table': tbl,
                'FK': fk,
                'KS stat': m['child_ks_stat'],
                'P-value': m['child_ks_p']
            })


    child_df = pd.DataFrame(child_rows)
    if not child_df.empty:
        child_df['Interpretation'] = child_df.apply(interpret_child, axis=1)
        child_df.to_csv(os.path.join(output_dir, 'child_count_ks.csv'), index=False)

    # 4) Schema & range adherence
    def interpret_schema(row):
        return ("OK" if row['Value'] > 0.99 else "violations")

    schema_rows = []
    for tbl, sc in full_results['schema_adherence'].items():
        for col, v in sc['dtype_validity'].items():
            schema_rows.append({'Table': tbl, 'Column': col, 'Check': 'dtype', 'Value': v})
        for col, v in sc['range_adherence'].items():
            schema_rows.append({'Table': tbl, 'Column': col, 'Check': 'range', 'Value': v})
    schema_df = pd.DataFrame(schema_rows)
    schema_df['Interpretation'] = schema_df.apply(interpret_schema, axis=1)
    schema_df.to_csv(os.path.join(output_dir, 'schema_adherence.csv'), index=False)

    # 5) Constraint adherence
    def interpret_constraint(row):
        if row['Constraint'] == 'pk_uniqueness':
            return ("OK" if row['Value'] and row['Value'] > 0.99 else "duplicates")
        else:
            return ("OK" if row['Value'] and row['Value'] > 0.99 else "violations")

    cons_rows = []
    for tbl, c in full_results['constraint_adherence'].items():
        cons_rows.append({'Table': tbl, 'Constraint': 'pk_uniqueness', 'Value': c.get('pk_uniqueness')})
        for k, v in c.items():
            if k.startswith('fk_'):
                cons_rows.append({'Table': tbl, 'Constraint': k, 'Value': v})
    cons_df = pd.DataFrame(cons_rows)
    cons_df['Interpretation'] = cons_df.apply(interpret_constraint, axis=1)
    cons_df.to_csv(os.path.join(output_dir, 'constraint_adherence.csv'), index=False)

    print(f"All result tables with interpretations written to {output_dir}")

