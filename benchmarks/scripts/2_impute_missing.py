from pathlib import Path
import os,sys
import pickle
import warnings
import importlib
import pandas as pd
import time
import yaml

from tqdm import tqdm
from migeclust.imputation import MultipleImputer

warnings.filterwarnings("ignore")

# ============================================================
# Define paths
# ============================================================
script_path = Path(__file__).resolve()
script_dir = script_path.parent

print("Script path:", script_path)
print("Script directory:", script_dir)

benchmark_dir = os.path.join(script_dir,"..")
data_dir = os.path.join(benchmark_dir,"data")

# ============================================================
# Set up configuration 
# ============================================================
yaml_path = os.path.join(script_dir,"../config/simulation_config.yaml")
benchmark_dir = os.path.join(script_dir,"..")

with open(yaml_path, "r") as f:
    config_data = yaml.safe_load(f)

ids = config_data['dataset_ids']
n_runs = config_data['n_runs']
md_param_grid = config_data['md_param_grid']

# ============================================================
# Pickle helpers 
# ============================================================
def read_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def write_pickle(var, path):
    with open(path, 'wb') as f:
       pickle.dump(var, f)


# ============================================================
# Create directories
# ============================================================
for id in ids:
    # create directory for dataset id in test_data/missing_data
    directory = os.path.join(data_dir,"imputed/"+str(id))
    if not os.path.exists(directory):
        os.makedirs(directory)
    # create directory for dataset id in test_data/missing_data/parameters
    for prop in md_param_grid['props']:
        for mf_proportion in md_param_grid['mf_proportions']:
            for mnar_proportion in md_param_grid['mnar_proportions']:
                directory = os.path.join(data_dir,"imputed/"+str(id)+"/"+str(prop)+"_"+str(mf_proportion)+"_"+str(mnar_proportion))
                if not os.path.exists(directory):
                    os.makedirs(directory)


# ============================================================
# MICE imputation
# ============================================================
for id in tqdm(ids,desc="Dataset processed"):
    for prop in md_param_grid['props']:
        for mf_proportion in md_param_grid['mf_proportions']:
            for mnar_proportion in md_param_grid['mnar_proportions']:
                
                # Load amputed data 
                directory = os.path.join(data_dir,"missing/"+str(id)+"/"+str(prop)+"_"+str(mf_proportion)+"_"+str(mnar_proportion))

                for seed in tqdm(range(n_runs), desc="Random seeds", leave=False):
                    missing_data_test = read_pickle(os.path.join(directory,"mdata_pipeline_"+str(seed)+".pkl"))

                    mice_imputer = MultipleImputer(
                        incomplete_data=missing_data_test.incomplete_data, 
                        seed = seed,
                        num_imputations=10,
                        mean_match_candidates=0,
                        mean_match_strategy = "Normal"
                    )
                    
                    mice_imputer.run_mice(iterations=2)
                    imputed_data = mice_imputer.get_multiple_imputations()
                    
                    directory_output = os.path.join(data_dir,"imputed/"+str(id)+"/"+str(prop)+"_"+str(mf_proportion)+"_"+str(mnar_proportion))
                    write_pickle(imputed_data, os.path.join(directory_output,"data_imputed_"+str(seed)+".pkl"))


