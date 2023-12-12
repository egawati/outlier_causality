# Outlier Causality in Data Streams

This repo contains data stream simulators to find point outlier causality explanations (anomaly root causes).
We propose Ocular and compare it to 4 existing algorithms: CausalRCA, CloudRanger, DyCause, and EasyRCA.
We conduct experiments on synthetic and real-world datasets. 

## Package Installation

We suggest to create a virtual environment to install the packages in this repository.

Create a virtual environment using Conda:
```
conda create --name outlier_causality --file environment.yml	
```

To Install Ocular:
```
cd ocular
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

