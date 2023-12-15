# Outlier Causality in Data Streams

This repo contains data stream simulators to find point outlier causality explanations (anomaly root causes).
We propose Ocular and compare it to 4 existing algorithms: CausalRCA, CloudRanger, DyCause, and EasyRCA.
We conduct experiments on synthetic and real-world datasets. 

## Package Installation

We suggest to create a virtual environment to install the packages in this repository.

Create a virtual environment using Conda:
```
conda config --add channels conda-forge
conda update -n base -c defaults conda
conda create --name outlier_causality python=3.8
```

First, let's install the base packages in Dev mode
```
cd bases/dowhy
pip install -e . 
cd ../EasyRCA
pip install -e .
```

Installing the rest of the packages
```
cd ../../ocular
pip install -e . 
cd ../detector/
pip install -e .
cd ../metrics
pip install -e .
cd ../metrics
pip install -e .
cd ../simulator_ocular
pip install -e .
cd ../simulator_cloudranger
pip install -e .
cd ../simulator_dycause
pip install -e .
cd ../simulator_easyrca
pip install -e .
cd ../simulator_gcm
pip install -e .
```

## Experiments
The datasets and modules to simulate them as data streams are in the `experiments` folder.
The simulation can be run after the packages are installed. 
For example, to run the simulation on varying the number of vertices on the Synthetic datasets with Ocular do the following:
```
cd experiments/ocular
python3 nodes.py
```

