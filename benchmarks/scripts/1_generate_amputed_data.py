from pathlib import Path
from tqdm import tqdm

import os,sys
import pickle
import pandas as pd
import importlib
import yaml

import benchutils.pipeline as pipeline

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

script_path = Path(__file__).resolve()
script_dir = script_path.parent

print("Script path:", script_path)
print("Script directory:", script_dir)

print("Loading configuration file...")

yaml_path = os.path.join(script_dir,"../config/simulation_config.yaml")
benchmark_dir = os.path.join(script_dir,"..")

with open(yaml_path, "r") as f:
    config_data = yaml.safe_load(f)

dataset_ids = config_data['dataset_ids']
n_runs = config_data['n_runs']

def load_dataset(id):
    with open(os.path.join(benchmark_dir,"data/raw/dataset_"+str(id)+".pkl"), "rb") as input:
        file = pickle.load(input)
    return file


loaded_data = {}
for id in dataset_ids: loaded_data[id] = load_dataset(id = id)

print(config_data)

md_param_grid = config_data['md_param_grid']

for id in dataset_ids:
    # create directory for dataset id in test_data/missing_data
    directory = "../../test_data/missing_data/"+str(id)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # create directory for dataset id in test_data/missing_data/parameters
    for prop in md_param_grid['props']:
        for mf_proportion in md_param_grid['mf_proportions']:
            for mnar_proportion in md_param_grid['mnar_proportions']:

                directory = os.path.join(
                    benchmark_dir,
                    "data/missing/"+str(id)+"/"+str(prop)+"_"+str(mf_proportion)+"_"+str(mnar_proportion)
                )
                if not os.path.exists(directory):
                    os.makedirs(directory)


for id in tqdm(dataset_ids):

    data_ = loaded_data[id]
    print(data_['X_complete'].shape)
    print(id)
    for prop in tqdm(md_param_grid['props']):
        for mf_proportion in tqdm(md_param_grid['mf_proportions']):
            for mnar_proportion in tqdm(md_param_grid['mnar_proportions']):
                
                directory = os.path.join(
                    benchmark_dir,
                    "data/missing/"+str(id)+"/"+str(prop)+"_"+str(mf_proportion)+"_"+str(mnar_proportion)
                )
                
                for seed in tqdm(range(n_runs)):
                    print(seed)
                    cfg = pipeline.DataPipelineConfig(seed=seed,
                                                      prop=prop,
                                                      mnar_freq=mnar_proportion,
                                                      mf_proportion=mf_proportion,
                                                      complete_data=data_['X_complete'],
                                                      num_classes=data_['num_classes'],
                                                      verbose=False,
                                                      target=data_['y_complete'])
                    pipeline_ = pipeline.PipelineDataGeneration(cfg)
                    pipeline_.run()
                    
                    directory = os.path.join(
                        benchmark_dir,
                        "data/missing/"+str(id)+"/"+str(prop)+"_"+str(mf_proportion)+"_"+str(mnar_proportion)
                    )
                    with open(directory+"/mdata_pipeline_"+str(seed)+".pkl", 'wb') as f:  
                        pickle.dump(pipeline_, f)