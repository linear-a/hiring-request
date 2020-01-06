import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('max_columns', 500)

df_report = pd.read_csv(
    './model_dev/dev_assets/outputs/single_variable_report.csv.gz',
    compression='gzip'
)

print(df_report.head())

df_report.sort_values(by='lm_auc', ascending=False).head(20)

# TEST CORRELATION
_df = pd.read_csv('./model_dev/dev_assets/data/1_clean_fe.csv.gz', compression='gzip')
print(_df.groupby('icce')['target'].sum())
print(_df[_df.icce == 'Cancer']['target'])

# STRONGEST LINEAR: provider_clinical_fte_sum_mav_5_vs_total TO target_60
target = 'target_30'
feature = 'provider_clinical_fte_sum_mav_3_vs_total'
df = _df.loc[:, [target, feature]]
df.dropna(inplace=True)

fig, ax = plt.subplots()

df_t0 = df[df.loc[:, target] == 0]
df_t1 = df[df.loc[:, target] == 1]

a_heights, a_bins = np.histogram(df_t0.loc[:, feature])
b_heights, b_bins = np.histogram(df_t1.loc[:, feature], bins=a_bins)

width = (a_bins[1] - a_bins[0])/3

ax.bar(a_bins[:-1], a_heights, width=width, facecolor='cornflowerblue')
ax.bar(b_bins[:-1]+width, b_heights, width=width, facecolor='seagreen')

df_t0[feature].hist()
df_t1[feature].hist()


print(df_t0[feature].describe())
print(df_t1[feature].describe())


wrvu = df_report[df_report.feature.str.contains('wrv')]
wrvu.sort_values(by='lm_auc', ascending=False).head(20)
