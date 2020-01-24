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

# LINEAR INTERPOLATION
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

# SET FEATURE LIST, AND WINDOWS
a = df.columns[6:]
window_list = [1, 2, 3, 5]

# LOOP THROUGH EACH FEATURE
# CREATE TOTALS, WINDOWS, COMPARISONS, TRENDS
for feature in a:
    df.loc[:, feature + '_total'] = df.groupby('icce')[feature] \
        .rolling(window=99999, min_periods=1) \
        .mean() \
        .reset_index(0, drop=True) \
        .fillna(0)

    for w in window_list:
        name = feature + '_mav_' + str(w)
        print(name)
        df.loc[:, name] = df.groupby('icce')[feature] \
            .rolling(window=w, min_periods=1) \
            .mean() \
            .reset_index(0, drop=True) \
            .fillna(0)

        df.loc[:, feature + '_mav_' + str(w) + '_vs_total'] = df.loc[:, name] - \
            df.loc[:, feature + '_total']

        df.loc[:, feature + '_mav_' + str(w) + '_trend'] = \
            np.where(df.loc[:, feature + '_mav_' + str(w) + '_vs_total'] > 0, 1,
                np.where(df.loc[:, feature + '_mav_' + str(w) + '_vs_total'] == 0, 0, -1))

df.to_csv(
    './model_dev/dev_assets/data/1_clean_fe.csv.gz',
    compression='gzip',
    index=False
)
