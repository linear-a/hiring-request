import numpy as np
import pandas as pd
from model_dev.dev_assets.sql import base_sql, prov_sql

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

project_name = 'musc-lineara'

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
print (list(m_fill))
print (list(years))

post = []
for y in years:
    y_str = str(y)
    post_pd = []
    for m in m_fill:
        x = y_str + m
        post_pd.append(x)
    post.append(post_pd)
bq_post_pd = [item for sublist in post for item in sublist]
print (bq_post_pd)
depts = list(df.ICCE.unique())
depts.extend(
    ['No ICCE Attribution',
     'UNKNOWN',
     'Interdisciplinary Hospital Staff',
     'Regional Health Network']
    )

d_df = pd.DataFrame(
    depts,
    columns=['ICCE']
    )

m_df = pd.DataFrame(
    bq_post_pd,
    columns=['POST_PD'],
    dtype='str'
    )

def cross_join(left, right):
    return (
       left.assign(key=1).merge(right.assign(key=1), on='key').drop('key', 1))

cj_df = cross_join(d_df, m_df)
print (cj_df)

result = pd.merge(cj_df,
                 df,
                 on=['POST_PD','ICCE'],
                 how='left')

print (result.TARGET.value_counts())
result['join'] = result.POST_PD.astype('int')
print (result)
print (result.TARGET.fillna('False').value_counts())
