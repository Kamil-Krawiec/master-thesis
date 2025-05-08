# ======================================
# DUAL DATASET GENERATOR COMPARISON SCRIPT
# ======================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.backends.backend_pdf import PdfPages

# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')

# Set Display Options
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 8})  # Set global font size to 8pt

# ------------------------
# 1. Define Paths
# ------------------------

output_path_both = 'analysis_results_both/'
os.makedirs(output_path_both, exist_ok=True)

datasets = {
    'northwind': {
        'dg_path': 'results-northwind-dataset/data_generator/',
        'mk_path': 'results-northwind-dataset/mockaroo/'
    },
    'basketball': {
        'dg_path': 'results-basketball-dataset/data_generator/',
        'mk_path': 'results-basketball-dataset/mockaroo/'
    }
}

file_names = [
    'child_count_ks.csv',
    'constraint_adherence.csv',
    'distance_metrics.csv',
    'schema_adherence.csv',
    'statistical_fidelity.csv'
]

# Helper to load and label data

def load_and_label(dataset_key):
    dg_path = datasets[dataset_key]['dg_path']
    mk_path = datasets[dataset_key]['mk_path']
    dg = {f.split('.')[0]: pd.read_csv(os.path.join(dg_path, f)).assign(generator='data_generator') for f in file_names}
    mk = {f.split('.')[0]: pd.read_csv(os.path.join(mk_path, f)).assign(generator='mockaroo') for f in file_names}
    merged = {k: pd.concat([dg[k], mk[k]], ignore_index=True) for k in dg}
    return merged

# Load both
merged_nw = load_and_label('northwind')
merged_bb = load_and_label('basketball')

# PDF report for both combined
pdf = PdfPages(os.path.join(output_path_both, 'dual_full_analysis_report.pdf'))

# Save helper
def save_plot(fig, filename):
    fig.savefig(os.path.join(output_path_both, filename), bbox_inches='tight')
    pdf.savefig(fig)
    plt.close(fig)

# ------------------------
# 2. Side-by-Side Comparisons
# ------------------------

def plot_dual_boxplot(df1, df2, metric_col, value_col, title, ylabel, filename):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    sns.boxplot(x=metric_col, y=value_col, data=df1, ax=axes[0])
    axes[0].set_title(f"Northwind: {title}")
    axes[0].set_xlabel('Generator')
    axes[0].set_ylabel(ylabel)
    axes[0].grid(True)

    sns.boxplot(x=metric_col, y=value_col, data=df2, ax=axes[1])
    axes[1].set_title(f"Basketball: {title}")
    axes[1].set_xlabel('Generator')
    axes[1].grid(True)

    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    save_plot(fig, filename)

# Child Count KS
child_nw = merged_nw['child_count_ks'][merged_nw['child_count_ks']['metric']=='child_ks_stat']
child_bb = merged_bb['child_count_ks'][merged_bb['child_count_ks']['metric']=='child_ks_stat']
plot_dual_boxplot(child_nw, child_bb, 'generator', 'value', 'Child Count KS Statistics', 'KS Statistic Value', 'child_count_ks_comparison.png')

# JS Divergence
js_nw = merged_nw['statistical_fidelity'][merged_nw['statistical_fidelity']['metric']=='js_divergence']
js_bb = merged_bb['statistical_fidelity'][merged_bb['statistical_fidelity']['metric']=='js_divergence']
plot_dual_boxplot(js_nw, js_bb, 'generator', 'value', 'JS Divergence', 'Divergence Value', 'js_divergence_comparison.png')

# Distance Metrics
from math import pi

def plot_dual_barplot(df1, df2, metric_col, value_col, title, ylabel, filename):
    df1 = df1.copy(); df1['normalized'] = df1[value_col] / df1.groupby(metric_col)[value_col].transform('max')
    df2 = df2.copy(); df2['normalized'] = df2[value_col] / df2.groupby(metric_col)[value_col].transform('max')
    avg1 = df1.groupby(['generator', metric_col])['normalized'].mean().reset_index()
    avg2 = df2.groupby(['generator', metric_col])['normalized'].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    sns.barplot(x=metric_col, y='normalized', hue='generator', data=avg1, ax=axes[0], errorbar=None)
    axes[0].set_title('Northwind: '+title); axes[0].set_ylabel(ylabel); axes[0].set_xlabel(metric_col.capitalize()); axes[0].grid(True)

    sns.barplot(x=metric_col, y='normalized', hue='generator', data=avg2, ax=axes[1], errorbar=None)
    axes[1].set_title('Basketball: '+title); axes[1].set_xlabel(metric_col.capitalize()); axes[1].grid(True)

    plt.suptitle(title, y=1.02); plt.tight_layout(); save_plot(fig, filename)

plot_dual_barplot(merged_nw['distance_metrics'], merged_bb['distance_metrics'], 'metric', 'value', 'Normalized Distance Metrics', 'Normalized Divergence', 'distance_metrics_comparison.png')

# Constraint & Schema Adherence
def plot_dual_simple_bar(df1, df2, title, ylabel, filename):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    sns.barplot(x='generator', y='value', data=df1, ax=axes[0], errorbar=None); axes[0].set_title('Northwind: '+title); axes[0].set_ylabel(ylabel); axes[0].set_xlabel('Generator'); axes[0].grid(True)
    sns.barplot(x='generator', y='value', data=df2, ax=axes[1], errorbar=None); axes[1].set_title('Basketball: '+title); axes[1].set_xlabel('Generator'); axes[1].grid(True)
    plt.suptitle(title, y=1.02); plt.tight_layout(); save_plot(fig, filename)

plot_dual_simple_bar(merged_nw['constraint_adherence'], merged_bb['constraint_adherence'], 'Constraint Adherence', 'Adherence Value', 'constraint_adherence_comparison.png')
plot_dual_simple_bar(merged_nw['schema_adherence'], merged_bb['schema_adherence'], 'Schema Adherence', 'Validity Score', 'schema_adherence_comparison.png')

# ------------------------
# 3. Example Sets Additional Analysis
# ------------------------
# Load the two example CSVs provided (Northwind data_generator)
sf_examples = pd.read_csv(os.path.join(datasets['northwind']['dg_path'], 'statistical_fidelity.csv'))
dm_examples = pd.read_csv(os.path.join(datasets['northwind']['dg_path'], 'distance_metrics.csv'))

# 3.1 Scatter: Distance Metrics vs JS Divergence
# Pivot distance metrics
dm_wide = dm_examples.pivot_table(index=['table','column'], columns='metric', values='value').reset_index()
# Extract JS from statistical fidelity
js_vals = sf_examples[sf_examples['metric']=='js_divergence'][['table','column','value']].rename(columns={'value':'js_divergence'})
# Merge for scatter
scatter_df = dm_wide.merge(js_vals, on=['table','column'])

fig = plt.figure(figsize=(8,6))
plt.scatter(scatter_df['tv'], scatter_df['js_divergence'], label='TV vs JS', marker='o')
plt.scatter(scatter_df['kl'], scatter_df['js_divergence'], label='KL vs JS', marker='x')
plt.title('Distance Metrics vs JS Divergence (Example Set)')
plt.xlabel('Distance Metric Value')
plt.ylabel('JS Divergence')
plt.legend(fontsize=6)
plt.grid(True)
save_plot(fig, 'example_scatter_distance_vs_js.png')

# 3.2 Correlation Heatmap of All Example Metrics
# Combine example sets
combined = pd.concat([
    dm_examples[['table','column','metric','value']],
    sf_examples[['table','column','metric','value']]
], ignore_index=True)
combined_wide = combined.pivot_table(index=['table','column'], columns='metric', values='value')

fig = plt.figure(figsize=(8,8))
sns.heatmap(combined_wide.corr(), annot=True, fmt=".2f", cbar=True)
plt.title('Correlation Between All Example Metrics')
save_plot(fig, 'example_metrics_correlation_heatmap.png')

# ------------------------
# Wrap Up
# ------------------------
pdf.close()
print("\n✅ Dual Dataset & Example Analysis Saved in 'analysis_results_both/' ✅")
