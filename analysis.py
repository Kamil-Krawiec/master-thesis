# ======================================
# FULL AUTOMATED GENERATOR COMPARISON SCRIPT
# ======================================

# Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

# Use non-interactive backend
matplotlib.use('Agg')

# Set Display Options
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")

# ------------------------
# 1. Define Paths
# ------------------------

# Base paths
data_generator_path = 'results-basketball-dataset/data_generator/'
mockaroo_path = 'results-basketball-dataset/mockaroo/'
output_path = 'analysis_results/'

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Define file names
file_names = [
    'child_count_ks.csv',
    'constraint_adherence.csv',
    'distance_metrics.csv',
    'schema_adherence.csv',
    'statistical_fidelity.csv'
]

# Load datasets
dg_data = {file.split('.')[0]: pd.read_csv(os.path.join(data_generator_path, file)) for file in file_names}
mk_data = {file.split('.')[0]: pd.read_csv(os.path.join(mockaroo_path, file)) for file in file_names}

# Label datasets
def label_dataset(df, generator_name):
    df['generator'] = generator_name
    return df

for key in dg_data:
    dg_data[key] = label_dataset(dg_data[key], 'data_generator')
    mk_data[key] = label_dataset(mk_data[key], 'mockaroo')

# Merge datasets
merged_data = {key: pd.concat([dg_data[key], mk_data[key]], ignore_index=True) for key in dg_data.keys()}

# Initialize PDF Report
pdf = PdfPages(os.path.join(output_path, 'full_analysis_report.pdf'))

# ------------------------
# 2. Analysis Functions
# ------------------------

def save_plot(fig, filename):
    fig.savefig(os.path.join(output_path, filename), bbox_inches='tight')
    pdf.savefig(fig)
    plt.close(fig)

def plot_boxplot(df, metric_col, value_col, title, ylabel, filename):
    fig = plt.figure(figsize=(12, 6))
    sns.boxplot(x=metric_col, y=value_col, data=df)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Generator')
    plt.grid(True)
    save_plot(fig, filename)

def plot_barplot(df, metric_col, value_col, title, ylabel, filename, ylim=None):
    fig = plt.figure(figsize=(10, 6))
    sns.barplot(x=metric_col, y=value_col, data=df, errorbar=None)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Generator')
    if ylim:
        plt.ylim(ylim)
    plt.grid(True)
    save_plot(fig, filename)

# ------------------------
# 3. Specific Analyses
# ------------------------

# 3.1 Child Count KS Analysis
child_ks_stat = merged_data['child_count_ks'][merged_data['child_count_ks']['metric'] == 'child_ks_stat']
plot_boxplot(child_ks_stat, 'generator', 'value', 'Child Count KS Statistics Comparison', 'KS Statistic Value', 'child_count_ks_stat_comparison.png')

# 3.2 Constraint Adherence Analysis
constraint_adherence = merged_data['constraint_adherence']
plot_barplot(constraint_adherence, 'generator', 'value', 'Constraint Adherence Comparison', 'Constraint Value', 'constraint_adherence_comparison.png', ylim=(0.9, 1.05))

# 3.3 Distance Metrics Analysis
distance_metrics = merged_data['distance_metrics']
fig = plt.figure(figsize=(16, 8))
sns.boxplot(x='generator', y='value', hue='metric', data=distance_metrics)
plt.title('Distance Metrics (TV, KL, JS) Comparison')
plt.ylabel('Divergence Value')
plt.xlabel('Generator')
plt.grid(True)
plt.legend(title='Metric')
save_plot(fig, 'distance_metrics_comparison.png')

# 3.4 Schema Adherence Analysis
schema_adherence = merged_data['schema_adherence']
plot_barplot(schema_adherence, 'generator', 'value', 'Schema Adherence Comparison', 'Schema Validity', 'schema_adherence_comparison.png', ylim=(0.9, 1.05))

# 3.5 Statistical Fidelity (JS Divergence) Analysis
statistical_fidelity = merged_data['statistical_fidelity']
js_div = statistical_fidelity[statistical_fidelity['metric'] == 'js_divergence']
plot_boxplot(js_div, 'generator', 'value', 'Statistical Fidelity - JS Divergence Comparison', 'JS Divergence Value', 'statistical_fidelity_js_divergence_comparison.png')

# ------------------------
# 4. Overall Score Comparison
# ------------------------

# Compute averages
score_summary = {
    'child_count_ks_avg': child_ks_stat.groupby('generator')['value'].mean(),
    'constraint_adherence_avg': constraint_adherence.groupby('generator')['value'].mean(),
    'distance_metrics_avg': distance_metrics.groupby('generator')['value'].mean(),
    'schema_adherence_avg': schema_adherence.groupby('generator')['value'].mean(),
    'js_divergence_avg': js_div.groupby('generator')['value'].mean(),
}

# Create DataFrame
score_df = pd.DataFrame(score_summary)
score_df.reset_index(inplace=True)

# Save scores as CSV
score_df.to_csv(os.path.join(output_path, 'generator_score_summary.csv'), index=False)

# Plot score comparison
for metric in score_df.columns[1:]:
    plot_barplot(score_df, 'generator', metric, f'Comparison: {metric.replace("_", " ").title()}', metric.replace("_", " ").title(), f'{metric}_comparison.png')

# ------------------------
# 5. Print Best Generators
# ------------------------

better_in = []
for metric in score_df.columns[1:]:
    better_gen = score_df.loc[score_df[metric].idxmin() if 'divergence' in metric or 'ks' in metric or 'distance' in metric else score_df[metric].idxmax(), 'generator']
    better_in.append((metric, better_gen))

# Save conclusions
with open(os.path.join(output_path, 'generator_comparison_conclusions.txt'), 'w') as f:
    f.write("=== 🧠 Overall Better Generator per Metric ===\n")
    for metric, gen in better_in:
        f.write(f"- {metric.replace('_', ' ').title()}: {gen}\n")

# Close PDF
pdf.close()

print("\n✅ FULL COMPARISON ANALYSIS SAVED SUCCESSFULLY! ✅")