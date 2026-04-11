import os
import glob
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 12, 
    'axes.titlesize': 14, 
    'axes.labelsize': 12, 
    'font.family': 'serif',
    'font.serif': ['Times New Roman']
})

BASE_PATH = './test_results'
OUT_DIR = os.path.join(BASE_PATH, 'plots_boxplot')
os.makedirs(OUT_DIR, exist_ok=True)

def get_metadata(filename):
    dataset = next((ds for ds in ['FIRMS-CHIRPS', 'VIIRS-CHIRPS', 'FIRMS', 'VIIRS'] if ds in filename), 'Unknown')
    
    variant = 'full'
    if 'model_chirps' in filename: variant = 'model_chirps'
    elif 'model' in filename: variant = 'model'
    elif 'reduced' in filename: variant = 'reduced'
    
    threshold = 'Unknown'
    for t in ['above-nominal_conf', 'high_conf', 'nominal_conf', '80', '90', '95']:
        if t in filename:
            threshold = t
            break
            
    return dataset, variant, threshold

def clean_label(label, dataset):
    mapping = {
        'nominal_conf': 'Nominal',
        'above-nominal_conf': 'Nominal/High',
        'high_conf': 'High',
        '80': '80%', '90': '90%', '95': '95%'
    }
    return mapping.get(label, label)

def generate_plots(data, sensor_type):
    if data.empty: return
    
    groups = list(data.groupby(['dataset', 'variant']))
    n_cols = 2
    n_rows = (len(groups) + 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 5 * n_rows))
    axes = axes.flatten() if len(groups) > 1 else [axes]

    for i, ((ds_name, var_name), group_df) in enumerate(groups):
        ax = axes[i]
        
        if sensor_type == 'VIIRS':
            order = [t for t in ['nominal_conf', 'above-nominal_conf', 'high_conf'] if t in group_df['threshold'].unique()]
        else:
            order = sorted(group_df['threshold'].unique())

        sns.boxplot(
            data=group_df, x='threshold', y='model_prob', order=order,
            palette='Set2', showmeans=True, ax=ax,
            meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"6"}
        )
        
        ax.set_xticklabels([clean_label(t, sensor_type) for t in order])
        ax.set_title(f"{ds_name.replace('-', ' with ')} ({var_name})")
        ax.set_xlabel('Confidence Threshold')
        ax.set_ylabel('Probability')
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        ax.text(-0.05, 1.05, chr(97 + i), transform=ax.transAxes, fontsize=14, fontweight='bold')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'{sensor_type}_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

# Data Ingestion
files = [f for f in glob.glob(f"{BASE_PATH}/*.csv") if "comparison" not in f]
data_list = []

for f in files:
    temp = pd.read_csv(f)
    ds, var, thr = get_metadata(os.path.basename(f))
    temp['dataset'], temp['variant'], temp['threshold'] = ds, var, thr
    temp['model_prob'] = pd.to_numeric(temp['model_prob'], errors='coerce')
    data_list.append(temp)

df = pd.concat(data_list, ignore_index=True)

# Process and Output
for sensor in ['FIRMS', 'VIIRS']:
    sensor_df = df[df['dataset'].str.contains(sensor)]
    print(f"\n--- {sensor} Statistics ---")
    print(sensor_df.groupby(['dataset', 'variant', 'threshold'])['model_prob'].agg(['mean', 'std', 'count']))
    generate_plots(sensor_df, sensor)
