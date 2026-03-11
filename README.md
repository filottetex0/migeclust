# MIGEClust

In `/src/`        folder --> Code for MIGEClust

In `/benchmarks/` folder --> Code for paper results replication 

We suggest to use a Conda environment:
Look <a href="https://www.anaconda.com/download" target="__blank">here</a> for a tutorial on how to install Conda/Miniconda. 

## Conda environment 

Use the following command to create the `migeclust` conda env. 
**N.B.:** in Windows, after installing conda/miniconda, you should use the "Anaconda prompt" to run the following commands.
```
conda create -n migeclust -c conda-forge \
    python=3.11 \
    pandas \
    scikit-learn \
    r-base \
    rpy2 \
    jupyter \
    tqdm \
    pip \
    seaborn
```
```
conda activate migeclust
pip install miceforest gower ucimlrepo scikit-posthocs
```
## Installing MixtureMissing for R 
<ins>This installation could be problematic, do it for paper results replication only.</ins>
```
conda activate migeclust
conda install -c conda-forge r-rspm r-Matrix r-MASS
R
install.packages("MixtureMissing")
```


## Install MIGEClust 
These steps assume you are working using a Unix-like terminal.
Steps 4-5 are useful if you want to replicate the code of the paper. 
1) Activate conda env:
```
conda activate migeclust
``` 
3) Clone the repository
```
git clone git@github.com:filottetex0/migeclust.git
```
or 
```
git clone https://github.com/filottetex0/migeclust
```
3) Install MIGEClust package
```
cd migeclust
pip install .
```
4) Clone and install helper libraries
```
cd benchmarks
mkdir -P libs
cd libs
git clone https://github.com/filottetex0/pyampute
git clone -b add_random_state https://github.com/filottetex0/kpod
pip install pyampute/.
pip install kpod/.
cd ..
```
5) Install benchmark utils:
In benchmarks folder
```
pip install .
```




























