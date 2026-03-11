from pathlib import Path
import os,sys
import pickle
import pandas as pd
import numpy as np
import importlib
import yaml
from tqdm import tqdm
from itertools import product
import time
from datetime import datetime

from gower import gower_matrix

import benchutils.clustering as clust_utils 
from benchutils.mica import compute_MICA
from benchutils.mixture_missing import run_mghm, run_mcnm
from benchutils.evaluation_metrics import external_metrics, internal_metrics

import migeclust.mige as migeClust


import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Define paths
# ============================================================
script_path = Path(__file__).resolve()
script_dir = script_path.parent

print("Script path:", script_path)
print("Script directory:", script_dir)

benchmark_dir = os.path.join(script_dir,"..")
output_dir = os.path.join(benchmark_dir,"output")
data_dir = os.path.join(benchmark_dir,"data")

os.makedirs(output_dir, exist_ok=True)

# ============================================================
# Configuration
# ============================================================
yaml_path = os.path.join(script_dir,"../config/simulation_config.yaml")
benchmark_dir = os.path.join(script_dir,"..")

with open(yaml_path, "r") as f:
    cfg = yaml.safe_load(f)

ids = cfg['dataset_ids']
n_runs = cfg['n_runs']
md_param_grid = cfg['md_param_grid']


# ============================================================
# Pickle loaders
# ============================================================
def read_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def write_pickle(var, path):
    with open(path, 'wb') as f:
       pickle.dump(var, f)





simulations_results = {}
configs = list(product(
    cfg['dataset_ids'],
    cfg['md_param_grid']['props'],
    cfg['md_param_grid']['mf_proportions'],
    cfg['md_param_grid']['mnar_proportions'],
    range(cfg['n_runs'])
))

TEST_DATASETS = [15,17,33,45,174,544]
#TEST_DATASETS = [544]


# ============================================================
# Simulation 
# ============================================================
filtered_configs = [cf for cf in configs if (cf[0] in TEST_DATASETS) and ((cf[4] == 0) or (cf[4] == 1)) and (cf[2] >= .75)]

for conf in tqdm(filtered_configs, position=0):

    dataset_id = conf[0]
    seed = conf[4]

    # test only datasets TEST_DATASETS

    md_config = str(conf[1])+"_"+str(conf[2])+"_"+str(conf[3])

    """
    Load data for simulation
    """
    complete_data_path = os.path.join(data_dir,"raw/dataset_"+str(dataset_id)+".pkl")
    missing_data_path = os.path.join(data_dir,"missing/"+str(dataset_id)+"/"+md_config+"/mdata_pipeline_"+str(seed)+".pkl")
    test_data_path = os.path.join(data_dir,"imputed/"+str(dataset_id)+"/"+md_config+"/data_imputed_"+str(seed)+".pkl")

    test_data_complete = read_pickle(complete_data_path)
    test_data_missing = read_pickle(missing_data_path)
    test_data = read_pickle(test_data_path)

    """
    Prepare data input
    """
    incomplete_data = test_data_missing.amputer.incomplete_dataset
    complete_data = test_data_complete['X_complete']
    true_labels = test_data_complete['y_complete'].values.flatten()
    cat_mask = test_data_complete['cat_mask']
    num_classes = test_data_complete['num_classes']
    multiple_imputed_data = test_data
    num_imputations = len(multiple_imputed_data)

    # hyperparameters
    consensuns_thresholds = [0.3,0.5,0.7]
    knn_numbers = [5,20,40] 
    projections = [0]

    # From configuration file
    # I also analyze projections
    # num_projections = cfg['mige_param']['num_projections']
    p_min = 1
    p_max = 1
    
    
    for proj, knn, co_thresh in tqdm(list(product(projections, knn_numbers, consensuns_thresholds)),position=1,leave=False):

        # compute labels
        mige_labels = migeClust.mige(
                        multiple_imputed_data,
                        n_clusters=num_classes,
                        cat_mask=cat_mask,
                        seed=seed,
                        p_min = p_min,
                        p_max = p_max,
                        num_projections = proj,
                        k_nn = knn,
                        co_threshold = co_thresh,
                        mutual=False
                    )

        # evaulate performance metrics
        try:
            int_metrics = internal_metrics(mige_labels, complete_data, cat_mask)
            ext_metrics = external_metrics(true_labels, mige_labels)
        except:
            int_metrics = np.nan
            ext_metrics = np.nan

        
        simulations_results[conf + (proj, knn, co_thresh)] = {}
        simulations_results[conf + (proj, knn, co_thresh)]['external_metrics'] = ext_metrics
        


# ============================================================
# Simulation - Dump
# ============================================================

timestamp = datetime.now().strftime("%d_%m_%Y")

write_pickle(
    simulations_results,
    os.path.join(
      output_dir,
      "sensitivity_results"+timestamp+".pkl"
    )
)

