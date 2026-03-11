"""
Script to fetch data from UCI repository
"""

import sys
import os
import pandas as pd
from ucimlrepo import fetch_ucirepo
import pickle
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm

script_path = Path(__file__).resolve()
script_dir = script_path.parent

print("Script path:", script_path)
print("Script directory:", script_dir)

print("Loading configuration file...")

yaml_path = os.path.join(script_dir,"../config/simulation_config.yaml")

with open(yaml_path, "r") as f:
    config_data = yaml.safe_load(f)

dataset_ids = config_data['dataset_ids']


def load_uciml(id: int = None, name: str = None, na_drop: bool = False):
    """
    Function to load data from the UCI Machine learning repository
    Args:
        id:
        name:
    Returns:
    """

    if id is None:
        dataset = fetch_ucirepo(name=name)
    else:
        dataset = fetch_ucirepo(id=id)

    # access data
    X = dataset.data.features
    if na_drop:
        X = X[pd.isna(X).sum()[~np.bool(pd.isna(X).sum().values)].index]

    # Remove duplicate columns (happened in ID 174)
    X = X.loc[:, ~X.columns.duplicated()]
    
    dataset.variables = dataset.variables.loc[~dataset.variables['name'].duplicated(),:]
    dataset.variables = dataset.variables.reset_index(drop=True)
    features = dataset.data.features.columns # bug : these are not sanitized

    # sanitize names
    X.columns = X.columns.str.replace(r'[^0-9a-zA-Z_]+', '_', regex=True)
    dataset.variables['name'] = dataset.variables['name'].str.replace(r'[^0-9a-zA-Z_]+', '_', regex=True)
    features = features.str.replace(r'[^0-9a-zA-Z_]+', '_', regex=True)

    y = dataset.data.targets

    cat_variables = dataset.variables.loc[(dataset.variables['type'] == 'Categorical') | (dataset.variables['type'] == 'Binary'), 'name']
    num_variables = dataset.variables.loc[(dataset.variables['type'] != 'Categorical') & (dataset.variables['type'] != 'Binary'), 'name']

    cat_variables = cat_variables[cat_variables.isin(features)]
    num_variables = num_variables[num_variables.isin(features)]

    obj_cols = X.select_dtypes(include='object').columns

    X = X.copy()
    X[obj_cols] = X[obj_cols].apply(lambda col: col.astype("category").cat.codes.astype(float)).copy()
    X[num_variables] = X[num_variables].astype("float64")

    cat_variables_in_X = cat_variables[cat_variables.isin(X.columns)]
    cat_mask = X.columns.isin(cat_variables_in_X)

    complete_rows_mask = ~(X.isnull().sum(axis = 1) > 0)
    
    X_complete = X.loc[complete_rows_mask,:].copy()
    X_complete = X_complete.apply(pd.to_numeric, errors="coerce")
    X_complete = X_complete.astype('float64')

    y_complete = y[complete_rows_mask].copy()

    res = {}
    res['X'] = X.copy()
    res['y'] = y
    res['cat_variables'] = cat_variables
    res['cat_variables_in_X'] = cat_variables_in_X
    res['cat_mask'] = cat_mask
    res['id'] = id

    res['X_complete'] = X_complete.copy()
    res['y_complete'] = y_complete

    res['num_classes'] = y_complete.value_counts().shape[0]


    return res


datasets = {}

for id in tqdm(dataset_ids):
    print(f"Collecting dataset {id}")
    datasets[id] = load_uciml(id=id)

# save the datasets into pickle file

raw_data_path = os.path.join(script_dir,"../data/raw")

for d in datasets:
    print(os.path.join(raw_data_path,"dataset_"+str(d)+".pkl"))
    with open(os.path.join(raw_data_path,"dataset_"+str(d)+".pkl"),"wb") as out:
        pickle.dump(datasets[d],out)

