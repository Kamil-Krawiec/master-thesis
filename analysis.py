#!/usr/bin/env python3
import os, glob
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
RESULTS_DIR = 'results-basketball-dataset'
GENERATORS  = ['data_generator', 'mockaroo']
OUTPUT_DIR  = 'analysis_output/basketball-dataset_scaled'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Interpretation rules for divergence
interpret_rules = {
    'tv': lambda v: 'noticeable divergence' if v >= 0.1 else 'high fidelity',
    'kl': lambda v: 'noticeable divergence' if v >= 0.1 else 'high fidelity',
    'js': lambda v: 'noticeable divergence' if v >= 0.1 else 'high fidelity',
    'wasserstein': lambda v: 'larger distance' if v >= 1 else 'small distance',
    'mmd': lambda v: 'larger MMD' if v >= 1e-3 else 'small MMD',
}

# --- LOAD ALL RESULTS ---
all_dfs = []
for gen in GENERATORS:
    path = os.path.join(RESULTS_DIR, gen, '*.csv')
    for fn in glob.glob(path):
        grp = os.path.splitext(os.path.basename(fn))[0]
        df = pd.read_csv(fn)
        df['generator']     = gen
        df['metric_group']  = grp
        all_dfs.append(df)
all_df = pd.concat(all_dfs, ignore_index=True)

# ------------------------------------------------------------------------
# 4.3 DISTANCE-BASED METRICS (filtered + scaled)
# ------------------------------------------------------------------------
dm = all_df[all_df.metric_group == 'distance_metrics'].copy()

# Safely convert column names and drop any id-like columns
dm['column'] = dm['column'].fillna('').astype(str)
dm = dm[~dm['column'].str.contains('id', case=False)]

# 1) Raw boxplots
plt.figure(figsize=(8,4))
dm.boxplot(column='value', by=['metric','generator'], rot=45)
plt.suptitle("Raw Distance Metrics by Generator")
plt.xlabel("Metric / Generator"); plt.ylabel("Value")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/distance_raw_box.png")
plt.close()

# 2) Scaled (min–max) boxplots
dm['scaled'] = dm.groupby('metric')['value'] \
    .transform(lambda x: (x - x.min())/(x.max() - x.min()))

plt.figure(figsize=(8,4))
dm.boxplot(column='scaled', by=['metric','generator'], rot=45)
plt.suptitle("Scaled Distance Metrics (0–1) by Generator")
plt.xlabel("Metric / Generator"); plt.ylabel("Scaled Value")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/distance_scaled_box.png")
plt.close()

# 3) Average scaled bar chart
avg_scaled = dm.groupby(['metric','generator'])['scaled'].mean().unstack()
ax = avg_scaled.plot(kind='bar', figsize=(6,3), rot=0)
ax.set_title("Average Scaled Distance per Metric")
ax.set_ylabel("Mean Scaled Value")
for bar in ax.patches:
    ax.annotate(f"{bar.get_height():.2f}",
                (bar.get_x()+bar.get_width()/2, bar.get_height()),
                ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/distance_avg_scaled.png")
plt.close()

# 4) Top‐5 divergent features per metric/generator
records = []
for gen in GENERATORS:
    sub = dm[dm.generator == gen]
    for m, rule in interpret_rules.items():
        flagged = sub[sub.metric == m].copy()
        flagged['status'] = flagged['value'].apply(rule)
        bad = flagged[flagged['status'] != 'high fidelity']
        top5 = bad.nlargest(5, 'value')[['table','column','metric','value']]
        top5['generator'] = gen
        records.append(top5)
top5_df = pd.concat(records, ignore_index=True)
print("\nTop 5 Divergent Features per Generator & Metric:")
print(top5_df.to_string(index=False, float_format="%.3f"))

# 5) Horizontal bar chart for top-5 JS divergences
js_worst = top5_df[top5_df.metric == 'js']
plt.figure(figsize=(6,3))
ax = js_worst.plot.barh(x='column', y='value', color=['#1f77b4','#ff7f0e'], legend=False)
plt.title("Top 5 JS Divergences")
plt.xlabel("JS Divergence")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/js_top5_h.png")
plt.close()

print(f"\nCharts saved to {OUTPUT_DIR}/")