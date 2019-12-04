## Import Standard Libraries
import numpy as np
import pandas as pd
import os
import gc
import glob
import scipy as sc

## Import Google-Specfic Libraries
import pandas_gbq as pdg
from google.oauth2 import service_account
from google.cloud import storage
from google.cloud import bigquery

## Import Additional Libraries
from model_dev.dev_assets import model_eval_functions

## Data Processing
import vtreat as vt
import wvpy.util

## Import Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xg
import skopt

## Test/Train Split
from sklearn.model_selection import train_test_split

## Set JSON Credentials;lo
credentials = 'g:/shared drives/sds/security/gcs/lineara-io-222305ef4e49.json'

## Set Project
project_name = 'musc-lineara'
dataset_name = 'SMP_FINAL'

## Set BigQuery Client
bq_client = bigquery.Client.from_service_account_json(credentials)

## Standard SQL
sql = """
WITH
  COUNTS AS (
  SELECT
    DATE(DATE_CREATED) AS DATE,
    CAST(FORMAT("%02d",(EXTRACT(MONTH
          FROM
            DATE(DATE_CREATED)))) AS STRING) AS MONTH,
    CAST(EXTRACT(YEAR
      FROM
        DATE(DATE_CREATED)) AS STRING) AS YEAR,
    ICCE,
    COUNT(REPLACEMENT) AS REPLACEMENT_CNTS
  FROM
    `musc-lineara.musc.custom_recruitment`
  GROUP BY
    DATE,
    MONTH,
    YEAR,
    ICCE
  ORDER BY
    DATE)
SELECT
  ICCE,
  CONCAT(YEAR, MONTH) AS POST_PD,
  TRUE AS TARGET
FROM
  COUNTS WHERE ICCE IS NOT NULL
GROUP BY
  ICCE,
  POST_PD
ORDER BY POST_PD
"""

## Read BigQuery into DataFrame
df = pdg.read_gbq(
    sql,
    project_id=project_name,
    dialect='standard')

print (df)

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
depts.extend(['No ICCE Attribution', 'UNKNOWN', 'Interdisciplinary Hospital Staff', 'Regional Health Network'])

d_df = pd.DataFrame(depts, columns=['ICCE'])

m_df = pd.DataFrame(bq_post_pd, columns=['POST_PD'], dtype='str')

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

prov_sql = """
WITH
  PROV_PROD AS (
  SELECT
    * EXCEPT(ICCE_NAME,
      PROVIDER_NPI,
      PROVIDER_NUMBER,
      NEW_PATIENT_APPTS_14_DAYS,
      NEW_PATIENT_APPTS_SCHEDULED,
      NEW_PATIENT_APPTS_TOTAL_LAG,
      COMPLETED_APPTS,
      POSSIBLE_SCHEDULED_APPTS,
      SLOTS_AVAILABLE),
    PROVIDER_NPI AS NPI,
    ICCE_NAME AS ICCE,
    COALESCE(SAFE_CAST(NEW_PATIENT_APPTS_14_DAYS AS FLOAT64),
      0.0) AS NEW_PATIENT_APPTS_14_DAYS,
    COALESCE(SAFE_CAST(NEW_PATIENT_APPTS_SCHEDULED AS FLOAT64),
      0.0) AS NEW_PATIENT_APPTS_SCHEDULED,
    COALESCE(SAFE_CAST(NEW_PATIENT_APPTS_TOTAL_LAG AS FLOAT64),
      0.0) AS NEW_PATIENT_APPTS_TOTAL_LAG,
    COALESCE(SAFE_CAST(COMPLETED_APPTS AS FLOAT64),
      0.0) AS COMPLETED_APPTS,
    COALESCE(SAFE_CAST(POSSIBLE_SCHEDULED_APPTS AS FLOAT64),
      0.0) AS POSSIBLE_SCHEDULED_APPTS,
    COALESCE(SAFE_CAST(SLOTS_AVAILABLE AS FLOAT64),
      0.0) AS SLOTS_AVAILABLE
  FROM
    `musc-lineara.musc.PROVIDER_PRODUCTIVITY_2`)
SELECT
  *
FROM
  PROV_PROD
  """
prov_df = pdg.read_gbq(
    prov_sql,
    project_id=project_name,
    dialect='standard')
print (prov_df.ICCE.value_counts())
print (list(prov_df.columns.values))
prov_df['join'] = prov_df.POST_PD.astype('int')

clean_nans={'TARGET':'False',
            'join':0,
            'PROVIDER_NAME':'UNKNOWN',
            'GENDER':'UNKNOWN',
            'PROVIDER_CLINICAL_FTE':0.0,
            'PROVIDER_TYPE':'Not Provided',
            'PROVIDER_SPECIALTY':'Not Provided',
            'PROVIDER_REPORTING_SPECIALTY':'Not Provided',
            'POST_PD_y':000000,
            'wRVUS':0.0,
            'UHCwRVUS':0.0,
            'NPI':0000000000,
            'NEW_PATIENT_APPTS_14_DAYS':0.0,
            'NEW_PATIENT_APPTS_SCHEDULED':0.0,
            'NEW_PATIENT_APPTS_TOTAL_LAG':0.0,
            'COMPLETED_APPTS':0.0,
            'POSSIBLE_SCHEDULED_APPTS':0.0,
            'SLOTS_AVAILABLE':0.0}

output = pd.merge(
    result,
                 prov_df,
                 on=['join','ICCE'], 
                 how='left'
                 )

variables = output.loc[:, output.columns != 'TARGET']
print (variables)
targets = output['TARGET'].fillna('False')
print (targets)
n = variables.shape[0]
targets.value_counts()

## Convert
dmap = {'True': 1, 'False': 0}
labels = targets.map(dmap).fillna(1)
print (labels.value_counts())

variables.drop(
    columns=['POST_PD_x',
             'POST_PD_y',
             'join',
             'ICCE',
             'PROVIDER_NAME',
             'GENDER',
             'NPI'],
    axis=0,
    inplace=True
    )
print (variables.columns.values)

## Split into Test/Train
train_features, test_features, train_labels, test_labels = train_test_split(variables, labels, test_size = 0.25, random_state = 42)

plan = vt.BinomialOutcomeTreatment(outcome_target=True)
cross_frame = plan.fit_transform(
    train_features,
    train_labels
    )

model_vars = np.asarray(plan.score_frame_['variable'][plan.score_frame_['recommended']])
len(model_vars)
cross_frame.dtypes

cross_sparse = sc.sparse.hstack([sc.sparse.csc_matrix(cross_frame[[vi]]) for vi in model_vars])

fd = xg.DMatrix(
    data=cross_sparse, 
    label=train_labels)

x_parameters = {'max_depth':4,
                'objective':'binary:logistic'}

cv = xg.cv(
    x_parameters,
    fd,
    num_boost_round=100,
    verbose_eval=False
    )

fitter = xg.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    objective='binary:logistic'
    )

model = fitter.fit(
    cross_sparse,
    train_labels
    )

test_processed = plan.transform(test_features)
print (test_processed)
pf_train = pd.DataFrame({'TARGET':train_labels})
pf_train['pred'] = model.predict_proba(cross_sparse)[:, 1]
wvpy.util.plot_roc(pf_train['pred'], pf_train['TARGET'], title='Model on Train')

test_sparse = sc.sparse.hstack([sc.sparse.csc_matrix(test_processed[[vi]]) for vi in model_vars])
pf = pd.DataFrame({'TARGET':test_labels})
pf['pred'] = model.predict_proba(test_sparse)[:, 1]
wvpy.util.plot_roc(pf['pred'], pf['TARGET'], title='Model on Test')

## Precision, Recall and F1
print(model_eval_functions.classification_eval(model, test_sparse, test_labels, cross_sparse, train_labels))
print (test_processed.columns)

## Evaluate
dt_feature_importances = pd.DataFrame(model.feature_importances_,
                                   index = test_processed.columns,
                                    columns=['importance']).sort_values('importance', ascending=False)
print (dt_feature_importances)