import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn import tree
from sklearn.metrics import log_loss, roc_curve, auc
pd.set_option('max_columns', 500)

df = pd.read_csv('./model_dev/dev_assets/data/1_clean_fe.csv.gz', compression='gzip')

print(df.shape)
print(df.target.mean())
print(df.target_30.mean())
print(df.target_60.mean())
print(df.target_90.mean())

# LOAD CLASSIFIERS
rpt = tree.DecisionTreeClassifier(max_depth=3)
glm_cv = LogisticRegressionCV(
    Cs=10,
    class_weight=None,
    cv=5,
    dual=False,
    fit_intercept=True,
    intercept_scaling=1.0,
    max_iter=100,
    multi_class='ovr',
    n_jobs=1,
    penalty='l2',
    random_state=None,
    refit=True,
    scoring=None,
    solver='lbfgs',
    tol=0.0001
)

single_mdl_report = pd.DataFrame(columns=[
    'feature',
    'target',
    'cardinality',
    'mean',
    'stddev',
    'min',
    'p5',
    'p25',
    'p50',
    'p75',
    'p95',
    'max',
    'kurtosis',
    'skew',
    'tree_auc',
    'tree_logloss',
    'min_prob_tree',
    'max_prob_tree',
    'lm_auc',
    'lm_logloss',
    'min_prob_lm',
    'max_prob_lm'
])

a = df.columns[6:]
y_list = ['target', 'target_30', 'target_60', 'target_90']

for col_nm in a:
    print('VARIABLE: ' + col_nm)

    for target in y_list:
        df_pre = df.loc[:, ['icce', col_nm, target]]
        df_pre.dropna(inplace=True)
        x = df_pre.loc[:, col_nm]
        x = np.array(x).reshape(-1, 1)
        y = df_pre.loc[:, target]

        print('FITTING DECISION TREE...')
        tree_mdl = rpt.fit(x, y)
        tree_probs = tree_mdl.predict_proba(x)[:, 1]
        tree_preds = tree_mdl.predict(x)

        tree_log_loss = log_loss(
            y_true=y,
            y_pred=tree_probs,
            labels=[0, 1]
        )

        fpr, tpr, thresholds = roc_curve(y, tree_probs, pos_label=1)
        tree_auc = auc(fpr, tpr)

        max_prob = tree_probs.max()
        max_val = df_pre.loc[df_pre[col_nm].idxmax()][1]

        min_prob = tree_probs.min()
        min_val = df_pre.loc[df_pre[col_nm].idxmin()][1]

        print('FITTING LINEAR MODEL...')
        linearmdl = glm_cv.fit(x, y)
        lm_probs = linearmdl.predict_proba(x)[:, 1]
        lm_preds = linearmdl.predict(x)

        lm_log_loss = log_loss(
            y_true=y,
            y_pred=lm_probs,
            labels=[0, 1]
        )

        fpr, tpr, thresholds = roc_curve(y, lm_probs, pos_label=1)
        lm_auc = auc(fpr, tpr)

        df_pre = df.loc[:, ['icce', col_nm]]

        max_prob_lm = lm_probs.max()
        max_val_lm = df_pre.loc[df_pre[col_nm].idxmax()][1]

        min_prob_lm = lm_probs.min()
        min_val_lm = df_pre.loc[df_pre[col_nm].idxmin()][1]

        cardinality = df_pre.loc[:, col_nm].nunique()
        col_mean, stddev, min, p25, p50, p75, max = df_pre.loc[:, col_nm].describe()[[1, 2, 3, 4, 5, 6, 7]]
        skew = df_pre.loc[:, col_nm].skew()
        kurt = df_pre.loc[:, col_nm].kurt()
        p5 = df_pre.loc[:, col_nm].quantile(0.05)
        p95 = df_pre.loc[:, col_nm].quantile(0.95)

        tmp = pd.DataFrame(columns=[
            'feature', 'target', 'cardinality', 'mean', 'stddev', 'min', 'p5', 'p25', 'p50',
            'p75', 'p95', 'max', 'kurtosis', 'skew',
            'tree_auc', 'tree_logloss',
            'min_prob_tree', 'max_prob_tree', 'lm_auc',
            'lm_logloss', 'min_prob_lm', 'max_prob_lm'
        ])

        tmp.loc[0] = [
            col_nm, target, cardinality, col_mean, stddev, min, p5, p25, p50,
            p75, p95, max, kurt, skew,
            tree_auc, tree_log_loss,
            min_prob, max_prob, lm_auc, lm_log_loss,
            min_prob_lm, max_prob_lm
        ]

        single_mdl_report = single_mdl_report.append(tmp, ignore_index=True)

    print(col_nm + ' COMPLETE. NEXT VAR...')
    print("")

single_mdl_report.to_csv(
    './model_dev/dev_assets/outputs/single_variable_report.csv.gz',
    compression='gzip',
    index=False
)
