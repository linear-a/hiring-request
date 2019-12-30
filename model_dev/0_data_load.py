import numpy as np
import pandas as pd
from model_dev.dev_assets.sql import base_sql, prov_sql
from model_dev.dev_assets.npi import npi_pull

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

project_name = 'musc-lineara'


def cross_join(left, right):
    return (left.assign(key=1).merge(right.assign(key=1), on='key').drop('key', 1))


df = pd.read_gbq(
    base_sql,
    project_id=project_name,
    dialect='standard'
)

print(df.shape)
print(df.head())

years = range(2017, 2020, 1)
months = range(1, 13, 1)
m_fill = []
for m in months:
    x = "{:02d}".format(m)
    m_fill.append(x)
print(list(m_fill))
print(list(years))

post = []
for y in years:
    y_str = str(y)
    post_pd = []
    for m in m_fill:
        x = y_str + m
        post_pd.append(x)
    post.append(post_pd)
bq_post_pd = [item for sublist in post for item in sublist]
print(bq_post_pd)
depts = list(df.icce.unique())
depts.extend([
    'No ICCE Attribution',
    'UNKNOWN',
    'Interdisciplinary Hospital Staff',
    'Regional Health Network'
])

d_df = pd.DataFrame(
    depts,
    columns=['icce']
)

m_df = pd.DataFrame(
    bq_post_pd,
    columns=['post_pd'],
    dtype='str'
)

cj_df = cross_join(d_df, m_df)
print(cj_df)

df = pd.merge(
    cj_df,
    df,
    on=['post_pd', 'icce'],
    how='left'
)

df.loc[:, 'target'].fillna(0, inplace=True)
print(df.target.value_counts())
df.loc[:, 'post_pd'] = df.loc[:, 'post_pd'].astype(int)

prov_df = pd.read_gbq(
    prov_sql,
    project_id=project_name,
    dialect='standard'
)
prov_df.columns = map(str.lower, prov_df.columns)

# npi_df = npi_pull(df=prov_df)
npi_df = pd.read_csv(
    './model_dev/dev_assets/data/md_api_output.csv.gz',
    compression='gzip'
)
print(npi_df.shape)

prov_df = pd.merge(
    prov_df,
    npi_df,
    on='npi',
    how='left'
)

prov_count = prov_df.groupby(['icce', 'post_pd']).size().reset_index()
prov_count.columns = ['icce', 'post_pd', 'num_providers']

prov_group = prov_df.groupby(['icce', 'post_pd']).agg({
    'provider_clinical_fte': ['mean', 'sum'],
    'wrvus': ['mean', 'sum'],
    'uhcwrvus': ['mean', 'sum'],
    'new_patient_appts_14_days': ['mean', 'sum'],
    'new_patient_appts_scheduled': ['mean', 'sum'],
    'new_patient_appts_total_lag': ['mean', 'sum'],
    'completed_appts': ['mean', 'sum'],
    'possible_scheduled_appts': ['mean', 'sum'],
    'slots_available': ['mean', 'sum'],
    'percentage_completed_appts_available': ['mean', 'sum'],
    'percentage_completed_slots_available': ['mean', 'sum']
}).reset_index()

prov_group.columns = ["_".join(x) for x in prov_group.columns.ravel()]
prov_group.rename(columns={'icce_': 'icce', 'post_pd_': 'post_pd'}, inplace=True)


prov_df_final = pd.merge(
    prov_count,
    prov_group,
    on=['icce', 'post_pd'],
    how='left'
)
## provider df needs to become single line per post_pd + icce before merging






df = pd.merge(
    df,
    prov_df,
    on=['post_pd', 'icce'],
    how='left'
)
