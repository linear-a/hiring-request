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

npi_df = npi_pull(df=prov_df)

prov_df = pd.merge(
    prov_df,
    npi_df,
    on='npi',
    how='left'
)


df = pd.merge(
    df,
    prov_df,
    on=['post_pd', 'icce'],
    how='left'
)
