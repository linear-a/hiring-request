import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

df = pd.read_csv(
    './model_dev/dev_assets/data/0_data_load.csv.gz',
    compression='gzip'
)

print(df.shape)

df.sort_values(by=['icce', 'post_pd'], inplace=True)
print(sum(df.post_pd_count.isnull()))
df = df.groupby('icce').apply(lambda group: group.interpolate(method='linear'))
print(sum(df.post_pd_count.isnull()))

# remove icce codes with all nulls
df = df[~df.icce.isin([
    'UNKNOWN',
    'Path & Lab',
    'Interdisciplinary Hospital Staff',
    'Digestive Health, Endocrine & Metabolism',
    'Children\'s & Women\'s',
    'Acute, Critical & Trauma'
])]

print(df.groupby('icce')['post_pd_count'].nunique())
print(df.target.value_counts())


