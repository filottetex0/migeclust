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
from benchutils.evaluation_metrics import *

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

# ============================================================
# Simulation
# ============================================================

simulations_results = {}

# all configurations
configs = list(product(
    cfg['dataset_ids'],
    cfg['md_param_grid']['props'],
    cfg['md_param_grid']['mf_proportions'],
    cfg['md_param_grid']['mnar_proportions'],
    range(cfg['n_runs'])
))


# ============================================================
# Simulation - Run
# ============================================================

configs = list(product(
    cfg['dataset_ids'],
    cfg['md_param_grid']['props'],
    cfg['md_param_grid']['mf_proportions'],
    cfg['md_param_grid']['mnar_proportions'],
    range(cfg['n_runs'])
))


for conf in tqdm(configs):

   dataset_id = conf[0]
   md_config = str(conf[1])+"_"+str(conf[2])+"_"+str(conf[3])
   seed = conf[4]
    
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


   """
   MIGE
   """

   num_projections = cfg['mige_param']['num_projections']
   p_min = cfg['mige_param']['p_min']
   p_max = cfg['mige_param']['p_max']
   knn = cfg['mige_param']['knn']
   
   num_imputations = len(multiple_imputed_data)
   co_threshold = 1/np.sqrt(num_imputations)



   mige_labels_no_proj = migeClust.mige(
                  multiple_imputed_data,
                  n_clusters=num_classes,
                  cat_mask=cat_mask,
                  seed=seed,
                  p_min = 1,
                  p_max = 1,
                  num_projections = 1,
                  k_nn = knn,
                  co_threshold = co_threshold,
                  mutual = False
               )
   
   mige_labels_proj = migeClust.mige(
                  multiple_imputed_data,
                  n_clusters=num_classes,
                  cat_mask=cat_mask,
                  seed=seed,
                  p_min = p_min,
                  p_max = p_max,
                  num_projections = num_projections,
                  k_nn = knn,
                  co_threshold = co_threshold,
                  mutual = False
               )
   """
   MIGE - Mutual
   """
   mige_labels_no_proj_mutual = migeClust.mige(
                  multiple_imputed_data,
                  n_clusters=num_classes,
                  cat_mask=cat_mask,
                  seed=seed,
                  p_min = 1,
                  p_max = 1,
                  num_projections = 1,
                  k_nn = knn,
                  co_threshold = co_threshold,
                  mutual = True
               )

   mige_labels_proj_mutual = migeClust.mige(
                  multiple_imputed_data,
                  n_clusters=num_classes,
                  cat_mask=cat_mask,
                  seed=seed,
                  p_min = p_min,
                  p_max = p_max,
                  num_projections = num_projections,
                  k_nn = knn,
                  co_threshold = co_threshold,
                  mutual = True
               )


   """
   MICA
   """
   mica_labels = compute_MICA(
      multiple_imputed_data,
      num_clusters=num_classes,
      seed=seed
   )


   """
   Kpod
   """
   kpod_labels = clust_utils.compute_kpod(
    incomplete_data,
    num_clusters=num_classes,
    seed=seed
   )


   """
   Mixture Missing
   """

   try:
    mghm_labels = run_mghm(
        incomplete_data,
        G=num_classes,
        seed=seed
        )
   except RuntimeError:
      mghm_labels = None 

   try:
      mcnm_labels = run_mcnm(
         incomplete_data,
         G=num_classes,
         seed=seed
         )
   except RuntimeError:
      mcnm_labels = None 

   """
   Naive methods
   """
   sc_si_knn_labels = clust_utils.compute_spectral_si_knn(
      incomplete_data,
      seed=seed,
      num_clusters=num_classes,
      cat_mask=cat_mask
   )

   sc_si_mi_labels = clust_utils.compute_spectral_si_mi(
      multiple_imputed_data,
      seed=seed,
      num_clusters=num_classes,
      cat_mask=cat_mask
   )

   km_si_knn_labels = clust_utils.compute_kmeans_si_knn(
      incomplete_data,
      num_clusters=num_classes,
      seed=seed
   )

   km_si_mi_labels = clust_utils.compute_kmeans_si_mi(
      multiple_imputed_data,
      num_clusters=num_classes,
      seed=seed
   )
    
   """
   CCA analyses
   """
   cca_spectral_labels = clust_utils.compute_spectral_complete(
      complete_data,
      cat_mask=cat_mask,
      num_clusters=num_classes,
      seed=seed
   )

   cca_kmeans_labels = clust_utils.compute_kmeans_complete(
      complete_data,
      num_clusters=num_classes,
      seed=seed
   )

   
   predicted_labels = {
      'mige_no_proj': mige_labels_no_proj,
      'mige_proj': mige_labels_proj,
      'mige_no_proj_mutual': mige_labels_no_proj_mutual,
      'mige_proj_mutual': mige_labels_proj_mutual,
      'mica': mica_labels,
      'kpod': kpod_labels,
      'mcnm': mcnm_labels,
      'mghm': mghm_labels,
      'sc_knn': sc_si_knn_labels,
      'sc_mi': sc_si_mi_labels,
      'km_knn': km_si_knn_labels,
      'km_si': km_si_mi_labels,
      'cca_spectral': cca_spectral_labels,
      'cca_kmeans': cca_kmeans_labels
   }

   int_metrics = dict.fromkeys(predicted_labels.keys())
   ext_metrics = dict.fromkeys(predicted_labels.keys())

   for method,comp in zip(predicted_labels.keys(),predicted_labels.values()):
    try:
        int_metrics[method] = internal_metrics(comp, complete_data, cat_mask)
        ext_metrics[method] = external_metrics(true_labels, comp)
    except:
        int_metrics[method] = np.nan
        ext_metrics[method] = np.nan

   simulations_results[conf] = {}
   simulations_results[conf]['internal_metrics'] = int_metrics
   simulations_results[conf]['external_metrics'] = ext_metrics

   #time.sleep(1)

# ============================================================
# Simulation - Dump
# ============================================================

timestamp = datetime.now().strftime("%d_%m_%Y")

write_pickle(
    simulations_results,
    os.path.join(
      output_dir,
      "simulation_results"+timestamp+".pkl"
    )
)






