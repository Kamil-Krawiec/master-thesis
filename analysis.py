# ======================================
# FULL AUTOMATED GENERATOR COMPARISON SCRIPT (UPDATED)
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
data_generator_path_northwind = 'results-northwind-dataset/data_generator/'
mockaroo_path_northwind = 'results-northwind-dataset/mockaroo/'
output_path_northwind = 'analysis_results-northwind/'

# Base paths
data_generator_path_basketball = 'results-basketball-dataset/data_generator/'
mockaroo_path_basketball = 'results-basketball-dataset/mockaroo/'
output_path_basketball = 'analysis_results-basketball/'

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

def plot_barplot(df, metric_col, value_col, title, ylabel, filename, ylim=None, add_labels=True, hue=None):
    fig = plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=metric_col, y=value_col, data=df, errorbar=None, hue=hue)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(metric_col.capitalize())
    if ylim:
        plt.ylim(ylim)
    plt.grid(True)

    if add_labels:
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', label_type='edge', padding=3, fontsize=9)

    save_plot(fig, filename)

def plot_boxplot(df, metric_col, value_col, title, ylabel, filename):
    fig = plt.figure(figsize=(12, 6))
    sns.boxplot(x=metric_col, y=value_col, data=df)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(metric_col.capitalize())
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
plot_barplot(constraint_adherence, 'generator', 'value', 'Constraint Adherence Comparison', 'Constraint Value', 'constraint_adherence_comparison.png', ylim=(0, 1.05))

# 3.3 Distance Metrics Analysis (NEW NORMALIZED VERSION)
distance_metrics = merged_data['distance_metrics']

# Normalize distance metrics
distance_metrics['normalized_value'] = distance_metrics['value'] / distance_metrics.groupby('metric')['value'].transform('max')

distance_avg = distance_metrics.groupby(['generator', 'metric'])['normalized_value'].mean().reset_index()

fig = plt.figure(figsize=(14, 7))
ax = sns.barplot(data=distance_avg, x='metric', y='normalized_value', hue='generator', errorbar=None)
plt.title('Normalized Distance Metrics (TV, KL, JS) Average Divergence Comparison')
plt.ylabel('Normalized Divergence Value')
plt.xlabel('Metric Type')
plt.grid(True)
plt.legend(title='Generator')

for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', label_type='edge', padding=3, fontsize=9)

save_plot(fig, 'distance_metrics_comparison.png')

# 3.4 Schema Adherence Analysis
schema_adherence = merged_data['schema_adherence']
plot_barplot(schema_adherence, 'generator', 'value', 'Schema Adherence Comparison', 'Schema Validity', 'schema_adherence_comparison.png', ylim=(0, 1.05))

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
    'distance_metrics_avg': distance_metrics.groupby('generator')['normalized_value'].mean(),
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
    plot_barplot(score_df, 'generator', metric, f'Comparison: {metric.replace("_", " ").title()}', metric.replace("_", " ").title(), f'{metric}_comparison.png', add_labels=True)

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
# ======================================
# 6. Extended Creative Visual Analysis (FINAL VERSION)
# ======================================

from math import pi

# --- 6.0 Prepare Normalized Data ---
# Normalize score dataframe for consistent scaling (0-1)
radar_df = score_df.copy()
for col in radar_df.columns[1:]:
    min_val = radar_df[col].min()
    max_val = radar_df[col].max()
    if max_val - min_val == 0:
        radar_df[col] = 0.5  # Avoid division by zero if no variance
    else:
        radar_df[col] = (radar_df[col] - min_val) / (max_val - min_val)

# Pivot original scores
pivot_df = score_df.set_index('generator')
norm_pivot = pivot_df.copy()
for col in norm_pivot.columns:
    min_val = norm_pivot[col].min()
    max_val = norm_pivot[col].max()
    norm_pivot[col] = (norm_pivot[col] - min_val) / (max_val - min_val)

# Normalize distance metrics for scaled boxenplot
distance_metrics_scaled = merged_data['distance_metrics'].copy()
distance_metrics_scaled['scaled_value'] = distance_metrics_scaled.groupby('metric')['value'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

# --- 6.1 Radar (Spider) Chart ---
categories = list(radar_df.columns[1:])
N = len(categories)

fig = plt.figure(figsize=(8,8))
ax = plt.subplot(111, polar=True)

for idx, row in radar_df.iterrows():
    values = row.drop('generator').tolist()
    values += values[:1]
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['generator'])
    ax.fill(angles, values, alpha=0.2)

plt.xticks([n / float(N) * 2 * pi for n in range(N)], categories, size=10)
plt.title('Overall Generator Performance (Normalized)', size=14)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
save_plot(fig, 'radar_generator_comparison.png')

# --- 6.2 Heatmap of Normalized Scores ---
fig = plt.figure(figsize=(10,6))
sns.heatmap(norm_pivot.T, annot=True, cmap='YlGnBu', fmt=".2f", cbar_kws={'label': 'Normalized Score'})
plt.title('Heatmap of Normalized Generator Scores Across Metrics', size=14)
save_plot(fig, 'heatmap_generator_scores.png')

# --- 6.3 Ranked Metrics Bar Chart (Per Metric) ---
ranking = []
for metric in score_df.columns[1:]:
    if 'divergence' in metric or 'ks' in metric or 'distance' in metric:
        best = score_df.loc[score_df[metric].idxmin()]
    else:
        best = score_df.loc[score_df[metric].idxmax()]
    ranking.append((metric, best['generator']))

ranking_df = pd.DataFrame(ranking, columns=['Metric', 'Best_Generator'])

fig = plt.figure(figsize=(14,7))
sns.countplot(data=ranking_df, x='Best_Generator', hue='Metric', palette='pastel', dodge=True)
plt.title('Wins Per Generator Across All Metrics', size=14)
plt.xlabel('Generator')
plt.ylabel('Number of Wins')
plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
save_plot(fig, 'wins_per_generator.png')

# --- 6.4 Difference Between Generators (Per Metric) ---
diff_df = pivot_df.diff().iloc[1]

fig = plt.figure(figsize=(10,6))
diff_df.sort_values().plot(kind='barh', color='slateblue')
plt.axvline(0, color='black')
plt.title('Difference (Mockaroo - Data Generator) Per Metric', size=14)
plt.xlabel('Difference Value')
plt.grid(True)
save_plot(fig, 'difference_between_generators.png')

# --- 6.5 Distance Metrics Spread (Boxen Plot, Normalized) ---
fig = plt.figure(figsize=(14,7))
sns.boxenplot(data=distance_metrics_scaled, x='metric', y='scaled_value', hue='generator', palette='coolwarm')
plt.title('Scaled Distance Metrics Value Spread (Boxen Plot)', size=14)
plt.ylabel('Normalized Divergence Value')
plt.xlabel('Metric')
plt.grid(True)
plt.legend(title='Generator')
plt.tight_layout()
save_plot(fig, 'boxenplot_distance_metrics_scaled.png')