# migeclust

## Conda environment 

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

pip install miceforest gower ucimlrepo scikit-posthocs

## Installing MixtureMissing for R 

activate conda env and go to R 

conda install -c conda-forge r-rspm
conda install -c conda-forge r-Matrix
conda install -c conda-forge r-MASS

R version (4.5.2)

R
>install.packages("MixtureMissing")


## Prepare for testing 

cd to migeclust 
pip install -e .

install libraries 

cd /path/to/repo/banchmarks/libs 

git clone git@github.com:simonetome/pyampute.git
pip install pyampute/.

**!kPOD need to update structure 

git clone -b add_random_state git@github.com:simonetome/kPOD.git
pip install kPOD/.

cd to benchmarks folder and 
pip install -e .



























