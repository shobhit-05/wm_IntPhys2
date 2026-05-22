import pandas as pd
import numpy as np

results = pd.read_csv('/gpfs/home/preiyalt/RQ2-benchmark-analysis/phase3_full_results.csv')
metadata = pd.read_csv('/gpfs/projects/infoseeking/preiyalt/Main/metadata.csv')

results = results[~results['scene'].astype(str).str.contains('_')]
results['scene'] = results['scene'].astype(int)

meta = metadata[metadata['type'] == '1_Possible'].copy()
meta['SceneIndex'] = meta['SceneIndex'].astype(int)

df = results.merge(meta[['SceneIndex', 'game_name', 'condition', 'env', 'Difficulty', 'Camera']], 
                   left_on='scene', right_on='SceneIndex', how='left')

print("=== OVERALL STATS ===")
print(f"Total scenes: {len(df)}")
print(f"Mean drop: {df['drop_pct'].mean():.1f}%")
print(f"Scenes with positive drop: {(df['drop_pct'] > 0).sum()}")
print(f"Scenes with negative drop: {(df['drop_pct'] < 0).sum()}")

print("\n=== BY CAMERA TYPE ===")
print(df.groupby('Camera')['drop_pct'].agg(['mean', 'count', 'std']).round(2))

print("\n=== BY DIFFICULTY ===")
print(df.groupby('Difficulty')['drop_pct'].agg(['mean', 'count', 'std']).round(2))

print("\n=== BY CONDITION ===")
print(df.groupby('condition')['drop_pct'].agg(['mean', 'count', 'std']).round(2))

print("\n=== TOP 10 BIGGEST DROPS ===")
print(df.nlargest(10, 'drop_pct')[['scene', 'drop_pct', 'game_name', 'Difficulty', 'Camera']].to_string())

print("\n=== TOP 10 BIGGEST INCREASES ===")
print(df.nsmallest(10, 'drop_pct')[['scene', 'drop_pct', 'game_name', 'Difficulty', 'Camera']].to_string())
