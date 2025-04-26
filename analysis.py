#!/usr/bin/env python3
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
RESULTS_DIR = 'results-northwind-dataset'
GENERATORS  = ['data_generator', 'mockaroo']
OUTPUT_DIR  = 'analysis_output/northwind-dataset'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LOAD ALL RESULTS ---
all_dfs = []
for gen in GENERATORS:
    d = os.path.join(RESULTS_DIR, gen)
    if not os.path.isdir(d):
        continue
    for fn in glob.glob(os.path.join(d, '*.csv')):
        grp = os.path.splitext(os.path.basename(fn))[0]
        df = pd.read_csv(fn)
        df['generator']    = gen
        df['metric_group'] = grp
        all_dfs.append(df)
all_df = pd.concat(all_dfs, ignore_index=True)

# --- HELPER TO ANNOTATE BARS ---
def annotate_bars(ax, fmt="{:.0f}"):
    for bar in ax.patches:
        h = bar.get_height()
        ax.annotate(fmt.format(h),
                    xy=(bar.get_x()+bar.get_width()/2, h),
                    xytext=(0,2), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
# ------------------------------------------------------------------------
# 4.2 STATISTICAL FIDELITY BOXPLOTS (replace your pivot code)
# ------------------------------------------------------------------------
sf = all_df[all_df.metric_group=='statistical_fidelity']

# KS statistic boxplot
plt.figure(figsize=(5,3))
sf[sf.metric=='ks_stat'].boxplot(column='value', by='generator', rot=0)
plt.suptitle("KS Statistic by Generator")
plt.xlabel(""); plt.ylabel("KS stat")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/sf_ks_stat_box.png")
plt.close()

# χ² p-value boxplot
plt.figure(figsize=(5,3))
sf[sf.metric=='chi2_p'].boxplot(column='value', by='generator', rot=0)
plt.suptitle("Chi² p-value by Generator")
plt.xlabel(""); plt.ylabel("p-value")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/sf_chi2_p_box.png")
plt.close()

# JS divergence boxplot
plt.figure(figsize=(5,3))
sf[sf.metric=='js_divergence'].boxplot(column='value', by='generator', rot=0)
plt.suptitle("JS Divergence by Generator")
plt.xlabel(""); plt.ylabel("JS divergence")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/sf_jsd_box.png")
plt.close()

# ------------------------------------------------------------------------
# 4.3 DISTANCE-BASED METRICS
#    • Boxplots of each metric
#    • Counts under threshold
# ------------------------------------------------------------------------
dm = all_df[all_df.metric_group=='distance_metrics']

# Boxplot of each distance metric by generator
plt.figure(figsize=(7,4))
dm.boxplot(column='value', by=['metric','generator'], rot=45)
plt.suptitle("Distance Metrics by Generator")
plt.xlabel("Metric, Generator")
plt.ylabel("Value")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/distance_metrics_box.png")
plt.close()

# Count under threshold for TV & JS
thresholds = {'tv':0.1, 'js':0.1}
rows = []
for gen in GENERATORS:
    sub = dm[dm.generator==gen]
    for metric, th in thresholds.items():
        total = sub[sub.metric==metric]['value'].count()
        good  = (sub[sub.metric==metric]['value'] < th).sum()
        rows.append((gen, metric, good, total, 100*good/total if total else 0))
dist_tbl = pd.DataFrame(rows, columns=['generator','metric','good','total','pct'])
print("\nDistance Metrics Under Threshold:")
print(dist_tbl.to_string(index=False, float_format="%.1f"))

# Vertical bar chart of counts under threshold
pivot = dist_tbl.pivot(index='metric', columns='generator', values='good')
plt.figure(figsize=(5,3))
ax = pivot.plot(kind='bar', rot=0)
ax.set_title("Count Under Threshold (TV & JS)")
ax.set_ylabel("Count")
annotate_bars(ax)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/distance_tv_js_good.png")
plt.close()

# ------------------------------------------------------------------------
# 4.4 SCHEMA & CONSTRAINT ADHERENCE
# ------------------------------------------------------------------------
sa = all_df[all_df.metric_group=='schema_adherence']
schema_tbl = sa.groupby(['generator','table'])['value'].mean().reset_index()
schema_tbl['status']=schema_tbl.value.apply(lambda v:'OK' if v>0.99 else 'Violations')
schema_cnt = schema_tbl.groupby(['generator','status']).size().unstack(fill_value=0)
schema_pct = (schema_cnt.div(schema_cnt.sum(axis=1), axis=0)*100).round(1)
print("\nSchema Adherence Counts:\n", schema_cnt)
print("\nSchema Adherence %:\n", schema_pct)

ax = schema_cnt.plot(kind='bar', figsize=(5,3), rot=0)
ax.set_title("Schema Adherence OK vs Violations")
ax.set_ylabel("Count")
annotate_bars(ax)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/schema_status.png")
plt.close()

ca = all_df[all_df.metric_group=='constraint_adherence']

# PK uniqueness
pk = ca[ca.constraint=='pk_uniqueness'].copy()
pk['status']=pk.value.apply(lambda v:'OK' if v>0.99 else 'Duplicates')
pk_cnt=pk.groupby(['generator','status']).size().unstack(fill_value=0)
pk_pct=(pk_cnt.div(pk_cnt.sum(axis=1), axis=0)*100).round(1)
print("\nPK Uniqueness Counts:\n", pk_cnt)
print("\nPK Uniqueness %:\n", pk_pct)

ax = pk_cnt.plot(kind='bar', figsize=(5,3), rot=0)
ax.set_title("PK Uniqueness OK vs Duplicates")
ax.set_ylabel("Count")
annotate_bars(ax)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/constraint_pk.png")
plt.close()

# FK validity
fk = ca[ca.constraint.str.startswith('fk_')].copy()
fk['status']=fk.value.apply(lambda v:'OK' if v>0.99 else 'Violations')
fk_cnt=fk.groupby(['generator','status']).size().unstack(fill_value=0)
fk_pct=(fk_cnt.div(fk_cnt.sum(axis=1), axis=0)*100).round(1)
print("\nFK Validity Counts:\n", fk_cnt)
print("\nFK Validity %:\n", fk_pct)

ax = fk_cnt.plot(kind='bar', figsize=(5,3), rot=0)
ax.set_title("FK Validity OK vs Violations")
ax.set_ylabel("Count")
annotate_bars(ax)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/constraint_fk.png")
plt.close()

print(f"\nTables printed above. Charts saved under '{OUTPUT_DIR}/'.")