import pandas as pd
import numpy as np
from npyi import npi
from pandas.io.json import json_normalize


def npi_pull(df=None):
    md_list = df.npi.unique().tolist()
    md_list = set(md_list)
    md_list = [x for x in md_list if str(x) != 'nan']
    md_list = [x for x in md_list if len(str(x)) >= 10]
    md_list = [int(x) for x in md_list]

    md_df = pd.DataFrame(columns=[
        'npi',
        'enumeration_date',
        'last_updated',
        'sole_proprietor',
        'specialty_desc',
        'primary'
    ])

    for md in md_list:
        print(md)
        try:
            response = npi.search(search_params={'number': md})
        except:
            print("missing: " + str(md))
            pass

        if [r for r in response.values()][0] == 0:
            print("no response for " + str(md))
            continue

        json_result = json_normalize(response, 'results')
        _npi = json_result.number[0]
        _desc = json_normalize(data=response['results'], record_path='taxonomies', meta_prefix='desc')

        _md_df = pd.DataFrame(columns=[
            'npi',
            'enumeration_date',
            'last_updated',
            'sole_proprietor',
            'specialty_desc',
            'primary'
        ])

        ed = json_result.loc[:, 'basic.enumeration_date'][0]
        updated = json_result.loc[:, 'basic.last_updated'][0]

        if 'basic.sole_proprietor' in json_result.columns:
            sp = json_result.loc[:, 'basic.sole_proprietor'][0]
        else:
            sp = np.nan
        ph_desc = _desc.desc[0]
        primary = _desc.primary[0]

        _md_df.loc[0] = [_npi, ed, updated, sp, ph_desc, primary]
        md_df = md_df.append(_md_df, ignore_index=True)

    md_df.to_csv(
        './model_dev/dev_assets/data/md_api_output.csv.gz',
        compression='gzip',
        index=False
    )

    return md_df
