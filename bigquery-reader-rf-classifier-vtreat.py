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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xg
import skopt

## Test/Train Split
from sklearn.model_selection import train_test_split

## Set Project
project_name = 'musc-lineara'
dataset_name = 'SMP_FINAL'

## Set BigQuery Client
bq_client = bigquery.Client()

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
      0.0) AS SLOTS_AVAILABLE,
    SAFE_DIVIDE(COALESCE(SAFE_CAST(COMPLETED_APPTS AS FLOAT64),
      0.0),
    (COALESCE(SAFE_CAST(POSSIBLE_SCHEDULED_APPTS AS FLOAT64),
      0.0))) AS PERCENTAGE_COMPLETED_APPTS_AVAILABLE,
    SAFE_DIVIDE(COALESCE(SAFE_CAST(COMPLETED_APPTS AS FLOAT64),
      0.0),
    (COALESCE(SAFE_CAST(SLOTS_AVAILABLE AS FLOAT64),
      0.0))) AS PERCENTAGE_COMPLETED_SLOTS_AVAILABLE
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
output.sort_values(
    inplace=True,
    by='POST_PD_y',
    na_position='first'
    )
print (output.columns.values)
variables = output.loc[:, output.columns != 'TARGET']
print (variables)
targets = output['TARGET'].fillna('False')
print (targets)
n = variables.shape[0]

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
             'NPI',
             'PROVIDER_REPORTING_SPECIALTY',
             'PROVIDER_SPECIALTY'],
    axis=0,
    inplace=True
    )
print (variables.columns.values)

## Data Prep for Train
plan = vt.BinomialOutcomeTreatment(
    outcome_target=True,
    params = vt.vtreat_parameters({
        'filter_to_recommended': False,
        'sparse_indicators': False
    }
    ))

cross_frame = plan.fit_transform(
    variables,
    labels
    )
cross_frame.dtypes
cross_frame.shape
print (cross_frame)
## Split into Test/Train
train_features, test_features, train_labels, test_labels = train_test_split(cross_frame, labels, test_size = 0.2, random_state = 42, shuffle=True)
model_vars = np.asarray(plan.score_frame_['variable'][plan.score_frame_['recommended']])

rf = xg.XGBClassifier(
    objective='binary:logistic'
    )

opt = skopt.BayesSearchCV(
    rf,
    {
        "max_depth": (2, 4),
        "n_estimators": (10, 100),
        "booster": ['gbtree', 'gblinear', 'dart']
    },
    n_iter=100,
    cv=4,
    scoring='f1',
    n_jobs=6
)

opt.fit(
    train_features,
    train_labels
    )
print("val. score: %s" % opt.best_score_)

tree_mdl = opt.best_estimator_

## Fit Best Estimator
tree_fit = tree_mdl.fit(
    train_features,
    train_labels
    )

## Classification predictions
rf_predictions = tree_fit.predict(train_features)

## Probabilities for AUC
rf_probs = tree_fit.predict_proba(train_features)[:, 1]

## ROC AUC
roc_value = roc_auc_score(
    train_labels,
    rf_probs
    )
print (roc_value)

## Precision, Recall and F1
print(model_eval_functions.classification_eval(
    tree_mdl,
    test_features,
    test_labels,
    train_features,
    train_labels
    ))

## Evaluate
dt_feature_importances = pd.DataFrame(
    tree_fit.feature_importances_,
    index = train_features.columns,
    columns=['importance']
    ).sort_values(
        'importance',
        ascending=False
        )
print (dt_feature_importances)

test_processed = plan.transform(test_features)
print (test_processed)

pf_train = pd.DataFrame({'TARGET':train_labels})
pf_train['pred'] = model.predict_proba(train_features)[:, 1]
wvpy.util.plot_roc(pf_train['pred'], pf_train['TARGET'], title='Model on Train')

test_sparse = sc.sparse.hstack([sc.sparse.csc_matrix(test_processed[[vi]]) for vi in model_vars])
pf = pd.DataFrame({'TARGET':test_labels})
pf['pred'] = model.predict_proba(test_features)[:, 1]
wvpy.util.plot_roc(pf['pred'], pf['TARGET'], title='Model on Test')