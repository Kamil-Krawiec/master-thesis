import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams.update({'font.size': 8})
sns.set(style="dark", palette="mako", color_codes=True)



output_dir = 'interpretation_stacked_by_generator/'
os.makedirs(output_dir, exist_ok=True)

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

def load_data(dataset, generator, fname):
    path = os.path.join(datasets[dataset][generator + '_path'], fname)
    df = pd.read_csv(path)
    df['dataset'] = dataset
    df['generator'] = 'Data Generator' if generator == 'dg' else 'Mockaroo'
    return df

# Load all data
all_data = {}
for fname in file_names:
    dfs = []
    for dataset in datasets:
        for generator in ['dg', 'mk']:
            df = load_data(dataset, generator, fname)
            dfs.append(df)
    all_data[fname.split('.')[0]] = pd.concat(dfs, ignore_index=True)

# Plot
for idx_chart, (key, df) in enumerate(all_data.items()):
    # Unique interpretations per figure
    unique_interpretations = df['interpretation'].dropna().unique()
    unique_interpretations = sorted(list(unique_interpretations))

    # New color palette per figure
    custom_palette = sns.color_palette("Set1", n_colors=len(unique_interpretations))
    color_map = {interp: custom_palette[idx] for idx, interp in enumerate(unique_interpretations)}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for i, dataset in enumerate(['northwind', 'basketball']):
        subset = df[df['dataset'] == dataset]
        counts = subset.groupby('generator')['interpretation'].value_counts(normalize=True).unstack(fill_value=0) * 100

        # Ensure all interpretations as columns
        for interp in unique_interpretations:
            if interp not in counts.columns:
                counts[interp] = 0
        counts = counts[unique_interpretations]

        counts.plot(kind='bar', stacked=True, ax=axes[i],
                    color=[color_map[c] for c in counts.columns],
                    legend=False)  # suppress default legends

        # Annotate
        for p in axes[i].patches:
            height = p.get_height()
            if height >= 1.5:
                x = p.get_x() + p.get_width() / 2
                y = p.get_y() + height / 2
                axes[i].text(x, y, f'{height:.1f}%', ha='center', va='center', fontsize=7)

        axes[i].set_title(f'{dataset.capitalize()} - {key.replace("_", " ").title()} (interpretation %)')
        axes[i].set_xlabel('Generator')
        axes[i].set_ylabel('Percentage')
        axes[i].set_ylim(0, 100)
        axes[i].tick_params(axis='x', rotation=0)
        axes[i].grid(axis='y')

    # Create shared legend ABOVE both charts
    handles = [plt.Rectangle((0,0),1,1, color=color_map[interp]) for interp in unique_interpretations]
    fig.legend(handles, unique_interpretations, title='Interpretation',
               loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=len(unique_interpretations))

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space on top for legend
    plt.savefig(os.path.join(output_dir, f'{key}_stacked_interpretation_by_generator.png'), bbox_inches='tight')
    plt.close(fig)

print(f"\n✅ Charts saved in '{output_dir}' — same palette per pair, legend centered above pair.")