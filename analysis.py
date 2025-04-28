#!/usr/bin/env python3
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# RESULTS_DIR = 'results-northwind-dataset'
# GENERATORS  = ['data_generator', 'mockaroo']
# OUTPUT_DIR  = 'analysis_output/northwind-dataset'
# os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- CONFIGURATION basketball ---
RESULTS_DIR = 'results-basketball-dataset'
GENERATORS  = ['data_generator', 'mockaroo']
OUTPUT_DIR  = 'analysis_output/basketball-dataset'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LOAD ALL RESULTS ---
all_dfs = []
for gen in GENERATORS:
    d = os.path.join(RESULTS_DIR, gen)
    if not os.path.isdir(d): continue
    for fn in glob.glob(os.path.join(d, '*.csv')):
        grp = os.path.splitext(os.path.basename(fn))[0]
        df = pd.read_csv(fn)
        df['generator']    = gen
        df['metric_group'] = grp
        all_dfs.append(df)
all_df = pd.concat(all_dfs, ignore_index=True)

# --- HELPERS ---
def annotate_bars(ax, fmt="{:.0f}"):
    # vertical bars
    for bar in ax.patches:
        h = bar.get_height()
        ax.annotate(fmt.format(h),
                    xy=(bar.get_x()+bar.get_width()/2, h),
                    xytext=(0,2), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

def annotate_hbars(ax, fmt="{:.0f}"):
    # horizontal bars
    for bar in ax.patches:
        w = bar.get_width()
        ax.annotate(fmt.format(w),
                    xy=(w, bar.get_y()+bar.get_height()/2),
                    xytext=(2,0), textcoords="offset points",
                    ha='left', va='center', fontsize=8)

# ------------------------------------------------------------------------
# 4.2 STATISTICAL FIDELITY BOXPLOTS
# ------------------------------------------------------------------------
sf = all_df[all_df.metric_group=='statistical_fidelity']
for metric,title in [('ks_stat',"KS Statistic"),('chi2_p',"Chi² p-value"),('js_divergence',"JS Divergence")]:
    plt.figure(figsize=(5,3))
    sf[sf.metric==metric].boxplot(column='value', by='generator', rot=0)
    plt.suptitle(f"{title} by Generator")
    plt.xlabel(""); plt.ylabel(title)
    plt.tight_layout()
    fn = metric.replace('_','') + "_box.png"
    plt.savefig(f"{OUTPUT_DIR}/{fn}")
    plt.close()

# ------------------------------------------------------------------------
# 4.2 PASS/FAIL & FIDELITY COUNTS (HORIZONTAL)
# ------------------------------------------------------------------------
# KS pass/fail
ks = sf[sf.metric=='p_value'].copy()
ks['pass'] = ks.value >= 0.05
ks_cnt = ks.groupby(['generator','pass']).size().unstack(fill_value=0)
print("\nKS Test Pass Counts:\n", ks_cnt)
ax = ks_cnt.plot(kind='barh', figsize=(5,3))
ax.set_title("KS Test: Pass vs Fail")
ax.set_xlabel("Count")
annotate_hbars(ax)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/sf_ks_pass_h.png")
plt.close()

# Chi² match/differ
chi = sf[sf.metric=='chi2_p'].copy()
chi['match'] = chi.value >= 0.05
chi_cnt = chi.groupby(['generator','match']).size().unstack(fill_value=0)
print("\nChi-Square Match Counts:\n", chi_cnt)
ax = chi_cnt.plot(kind='barh', figsize=(5,3))
ax.set_title("Chi-Square: Match vs Differ")
ax.set_xlabel("Count")
annotate_hbars(ax)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/sf_chi2_match_h.png")
plt.close()

# JS high/low fidelity
js = sf[sf.metric=='js_divergence'].copy()
js['fidelity'] = js.value.apply(lambda v: 'high' if v<0.1 else 'low')
js_cnt = js.groupby(['generator','fidelity']).size().unstack(fill_value=0)
print("\nJS Divergence Counts:\n", js_cnt)
ax = js_cnt.plot(kind='barh', figsize=(5,3))
ax.set_title("JS Divergence: High vs Low Fidelity")
ax.set_xlabel("Count")
annotate_hbars(ax)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/sf_js_fidelity_h.png")
plt.close()

# ------------------------------------------------------------------------
# 4.3 DISTANCE-BASED METRICS
# ------------------------------------------------------------------------
dm = all_df[all_df.metric_group=='distance_metrics']

# Boxplot of all distance metrics
plt.figure(figsize=(7,4))
dm.boxplot(column='value', by=['metric','generator'], rot=45)
plt.suptitle("Distance Metrics by Generator")
plt.xlabel("Metric & Generator"); plt.ylabel("Value")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/distance_metrics_box.png")
plt.close()

# Count under threshold (TV & JS) horizontally
thresh={'tv':0.1,'js':0.1}
rows=[]
for gen in GENERATORS:
    sub=dm[dm.generator==gen]
    for m,t in thresh.items():
        total=sub[sub.metric==m]['value'].count()
        good=(sub[sub.metric==m]['value']<t).sum()
        rows.append((gen,m,good,total))
dist_tbl=pd.DataFrame(rows,columns=['generator','metric','good','total'])
print("\nDistance Under Threshold:\n", dist_tbl)
pivot=dist_tbl.pivot(index='metric',columns='generator',values='good')
ax=pivot.plot(kind='barh',figsize=(5,3))
ax.set_title("Count Under Threshold (TV & JS)")
ax.set_xlabel("Count")
annotate_hbars(ax)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/distance_tv_js_good_h.png")
plt.close()

# ------------------------------------------------------------------------
# 4.4 SCHEMA & CONSTRAINT ADHERENCE
# ------------------------------------------------------------------------
# Schema adherence
sa = all_df[all_df.metric_group=='schema_adherence']
st=sa.groupby(['generator','table'])['value'].mean().reset_index()
st['status']=st.value.apply(lambda v:'OK' if v>0.99 else 'Violations')
st_cnt=st.groupby(['generator','status']).size().unstack(fill_value=0)
print("\nSchema Adherence Counts:\n", st_cnt)
ax=st_cnt.plot(kind='barh',figsize=(5,3))
ax.set_title("Schema Adherence Status")
ax.set_xlabel("Count")
annotate_hbars(ax)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/schema_status_h.png")
plt.close()

# PK uniqueness
ca = all_df[all_df.metric_group=='constraint_adherence']
pk=ca[ca.constraint=='pk_uniqueness'].copy()
pk['status']=pk.value.apply(lambda v:'OK' if v>0.99 else 'Duplicates')
pk_cnt=pk.groupby(['generator','status']).size().unstack(fill_value=0)
print("\nPK Uniqueness Counts:\n", pk_cnt)
ax=pk_cnt.plot(kind='barh',figsize=(5,3))
ax.set_title("PK Uniqueness Status")
ax.set_xlabel("Count")
annotate_hbars(ax)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/constraint_pk_h.png")
plt.close()

# FK validity
fk=ca[ca.constraint.str.startswith('fk_')].copy()
fk['status']=fk.value.apply(lambda v:'OK' if v>0.99 else 'Violations')
fk_cnt=fk.groupby(['generator','status']).size().unstack(fill_value=0)
print("\nFK Validity Counts:\n", fk_cnt)
ax=fk_cnt.plot(kind='barh',figsize=(5,3))
ax.set_title("FK Validity Status")
ax.set_xlabel("Count")
annotate_hbars(ax)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/constraint_fk_h.png")
plt.close()

print(f"\nTables printed above. Charts saved under '{OUTPUT_DIR}/'.")