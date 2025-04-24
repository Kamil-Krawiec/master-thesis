# metrics_researcher.py

"""
Research-grade metrics for evaluating synthetic relational data.

Derived from:
- IRG: Incremental Relational Generator [1]  [oai_citation:7‡arXiv](https://arxiv.org/abs/2312.15187?utm_source=chatgpt.com)
- SQLSynthGen: Differentially-Private SQL Synthesizer [2]  [oai_citation:8‡ResearchGate](https://www.researchgate.net/publication/388722401_SQLSynthGen_Generating_Synthetic_Data_for_Healthcare_Databases?utm_source=chatgpt.com)
- Synthetic Data Generation for Enterprise DBMS [3]  [oai_citation:9‡IEEE Computer Society](https://www.computer.org/csdl/proceedings-article/icde/2023/222700d585/1PByIKpOQow?utm_source=chatgpt.com)

Metrics:
  4.2 Statistical fidelity (KS, chi-square, child-count KS) [4][5]  [oai_citation:10‡Wikipedia](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test?utm_source=chatgpt.com) [oai_citation:11‡arXiv](https://arxiv.org/pdf/2312.15187?utm_source=chatgpt.com)
  4.3 Distance-based (TV, KL, JS, Wasserstein, MMD) [6][7][8]  [oai_citation:12‡Wikipedia](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence?utm_source=chatgpt.com) [oai_citation:13‡Wikipedia](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence?utm_source=chatgpt.com) [oai_citation:14‡Wikipedia](https://en.wikipedia.org/wiki/Kernel_embedding_of_distributions?utm_source=chatgpt.com)
  4.4 Schema & constraint adherence (dtype, range, PK, FK) [1][2]  [oai_citation:15‡arXiv](https://arxiv.org/abs/2312.15187?utm_source=chatgpt.com) [oai_citation:16‡ResearchGate](https://www.researchgate.net/publication/388722401_SQLSynthGen_Generating_Synthetic_Data_for_Healthcare_Databases?utm_source=chatgpt.com)
  4.1 Performance & domain checks (generation time, dialect coverage) [3]  [oai_citation:17‡IEEE Computer Society](https://www.computer.org/csdl/proceedings-article/icde/2023/222700d585/1PByIKpOQow?utm_source=chatgpt.com)
"""

import numpy as np
import pandas as pd
import time
from scipy.stats import ks_2samp, chi2_contingency, wasserstein_distance, entropy
from sklearn.metrics.pairwise import rbf_kernel


def compute_statistical_fidelity(real: pd.Series, synth: pd.Series):
    """
    - Numeric → two-sample KS test (nonparametric) [4]  [oai_citation:18‡Wikipedia](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test?utm_source=chatgpt.com)
    - Categorical → chi-square (with Laplace smoothing) + JS divergence [5][6]  [oai_citation:19‡arXiv](https://arxiv.org/pdf/2312.15187?utm_source=chatgpt.com) [oai_citation:20‡Wikipedia](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence?utm_source=chatgpt.com)
    """
    real = real.dropna()
    synth = synth.dropna()

    # Try KS on numeric
    real_num = pd.to_numeric(real, errors='coerce').dropna()
    synth_num = pd.to_numeric(synth, errors='coerce').dropna()
    if len(real_num) and len(synth_num):
        stat, p = ks_2samp(real_num, synth_num)
        return {'test':'ks', 'ks_stat':stat, 'p_value':p}

    # Categorical fallback
    r_counts = real.astype(str).value_counts()
    s_counts = synth.astype(str).value_counts()
    idx = r_counts.index.union(s_counts.index)
    # Laplace smoothing to avoid zero-expected frequencies
    r = r_counts.reindex(idx, fill_value=0).values + 1
    s = s_counts.reindex(idx, fill_value=0).values + 1

    # Chi-square
    obs = np.stack([r, s], axis=1)
    chi2, chi2_p, _, _ = chi2_contingency(obs)
    # JS divergence
    p_norm = r / r.sum()
    q_norm = s / s.sum()
    m = 0.5 * (p_norm + q_norm)
    jsd = 0.5 * (entropy(p_norm, m) + entropy(q_norm, m))

    return {
        'test':'chi2_jsd',
        'chi2_stat':chi2, 'chi2_p':chi2_p,
        'js_divergence':jsd
    }


def compute_child_count_ks(real_child: pd.DataFrame,
                           synth_child: pd.DataFrame,
                           fk: str):
    """
    KS on distribution of number of child rows per parent key [1]  [oai_citation:21‡arXiv](https://arxiv.org/abs/2312.15187?utm_source=chatgpt.com).
    """
    rc = real_child.groupby(fk).size()
    sc = synth_child.groupby(fk).size()
    stat, p = ks_2samp(rc, sc)
    return {'child_ks_stat':stat, 'child_ks_p':p}


def compute_distance_metrics(real: pd.Series, synth: pd.Series):
    """
    Compute:
      - Total Variation Distance (TV)
      - Kullback–Leibler divergence (KL)
      - Jensen–Shannon divergence (JS)
      - Wasserstein distance (numeric only)
      - Maximum Mean Discrepancy (MMD, numeric only)
    Returns a dict; non‐numeric metrics are set to None.
    """
    # Build discrete distributions for any dtype
    real_counts = real.fillna('NA').astype(str).value_counts(normalize=True)
    synth_counts = synth.fillna('NA').astype(str).value_counts(normalize=True)
    idx = real_counts.index.union(synth_counts.index)
    p = real_counts.reindex(idx, fill_value=0).values
    q = synth_counts.reindex(idx, fill_value=0).values

    # Total Variation
    tv = 0.5 * np.sum(np.abs(p - q))
    # KL divergence with smoothing
    kl = entropy(p + 1e-12, q + 1e-12)
    # JS divergence
    m = 0.5 * (p + q)
    js = 0.5 * (entropy(p + 1e-12, m) + entropy(q + 1e-12, m))

    # Initialize numeric-only metrics
    wass = None
    mmd  = None

    # Check if column is numeric
    if pd.api.types.is_numeric_dtype(real) and pd.api.types.is_numeric_dtype(synth):
        # Convert and drop NaNs
        real_num = pd.to_numeric(real, errors='coerce').dropna().astype(float)
        synth_num = pd.to_numeric(synth, errors='coerce').dropna().astype(float)

        if len(real_num) and len(synth_num):
            # Wasserstein distance
            wass = wasserstein_distance(real_num, synth_num)

            # MMD via RBF kernel
            X = real_num.values.reshape(-1, 1)
            Y = synth_num.values.reshape(-1, 1)
            Kxx = rbf_kernel(X, X)
            Kyy = rbf_kernel(Y, Y)
            Kxy = rbf_kernel(X, Y)
            mmd = (Kxx.sum() / (len(X)**2)
                   + Kyy.sum() / (len(Y)**2)
                   - 2 * Kxy.sum() / (len(X) * len(Y)))

    return {
        'tv': tv,
        'kl': kl,
        'js': js,
        'wasserstein': wass,
        'mmd': mmd
    }

def compute_schema_adherence(df: pd.DataFrame,
                             dtype_map: dict,
                             range_map: dict):
    """
    Check:
      - Data type validity per column
      - Range adherence (min/max)
    """
    # Data type validity
    dtype_results = {}
    for col, dt in dtype_map.items():
        series = df[col]
        if dt.startswith('int'):
            # coerce to float, then test integer‐ness
            nums = pd.to_numeric(series, errors='coerce').dropna().astype(float)
            valid = nums.apply(lambda x: x.is_integer())
        elif dt in ('float','real'):
            valid = pd.to_numeric(series, errors='coerce').notna()
        elif dt in ('date','timestamp'):
            valid = pd.to_datetime(series, errors='coerce').notna()
        else:
            valid = series.notna()
        dtype_results[col] = valid.mean()

    # Range adherence
    range_results = {}
    for col, (lo, hi) in range_map.items():
        # only check numeric ranges
        nums = pd.to_numeric(df[col], errors='coerce').dropna().astype(float)
        in_range = nums.between(lo, hi) if lo is not None and hi is not None else pd.Series([True]*len(nums))
        range_results[col] = in_range.mean()

    return {'dtype_validity': dtype_results, 'range_adherence': range_results}


def compute_constraint_adherence(synth: dict, metadata: dict):
    """
    Check:
      - PK uniqueness [1]  [oai_citation:29‡arXiv](https://arxiv.org/abs/2312.15187?utm_source=chatgpt.com)
      - FK integrity [1]  [oai_citation:30‡arXiv](https://arxiv.org/abs/2312.15187?utm_source=chatgpt.com)
    metadata: {table: {'pk': [...], 'fks':[{'col','parent','pk'}]}}
    """
    results = {}
    for table, df in synth.items():
        meta = metadata[table]
        # PK uniqueness
        pk_cols = meta['pk']
        unique = df.drop_duplicates(pk_cols).shape[0] / len(df) if len(df) else 0
        results.setdefault(table, {})['pk_uniqueness'] = unique

        # FK integrity
        for fk in meta['fks']:
            parent_df = synth[fk['parent']]
            valid = df[fk['col']].isin(parent_df[fk['pk']])
            results[table][f"fk_{fk['col']}"] = valid.mean()

    return results


def interpret_results(metrics: dict):
    """
    Print human-readable interpretations:
      - KS p > 0.05 ⇒ no significant distribution shift [4]  [oai_citation:31‡Wikipedia](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test?utm_source=chatgpt.com)
      - JS/TV small ⇒ high fidelity [6][7]  [oai_citation:32‡Wikipedia](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence?utm_source=chatgpt.com) [oai_citation:33‡Wikipedia](https://en.wikipedia.org/wiki/F-divergence?utm_source=chatgpt.com)
      - MMD small ⇒ distributions close in RKHS [10]  [oai_citation:34‡Wikipedia](https://en.wikipedia.org/wiki/Kernel_embedding_of_distributions?utm_source=chatgpt.com)
      - PK uniqueness ≈1.0 and FK integrity ≈1.0 ⇒ schema respected [1]  [oai_citation:35‡arXiv](https://arxiv.org/abs/2312.15187?utm_source=chatgpt.com)
    """
    print("=== Statistical Fidelity ===")
    for tbl, cols in metrics['statistical'].items():
        print(f"\nTable: {tbl}")
        for col, m in cols.items():
            if m.get('test')=='ks':
                p = m['p_value']
                print(f"  • {col}: KS p-value={p:.3f} → "
                      + ("PASSED" if p>0.05 else "SHIFT DETECTED"))
            else:
                js = m['js_divergence']
                print(f"  • {col}: JS divergence={js:.4f} (0=perfect)")

    print("\n=== Constraint Adherence ===")
    for tbl, vals in metrics['constraints'].items():
        print(f"\nTable: {tbl}")
        print(f"  • PK uniqueness: {vals['pk_uniqueness']:.3f}")
        for k,v in vals.items():
            if k.startswith('fk_'):
                print(f"  • {k}: {v:.3f}")