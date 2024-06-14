# Fair-kFood
Offical codebase for our work "Towards Fairness in Online Service with k Servers and its Applications to Fair Food Delivery" (AAAI 2024).


# Getting Started
---
These instructions will get you a copy of the project up and running on your local machine.

## Prerequisites
The entire codebase is written in Python 3.8.1

## Installation
Setup a conda environment which includes packages required to run evaluation scripts:
```
conda env create -f environment.yaml
conda activate fair_kserver
```
Additionally, access to Gurobi Optimizer is required. Access to Gurobi's academic (free!) license can be requested [here]().

## Dataset
* To get access to the food-delivery dataset, please follow the procedure defined [here]().
* The scripts to generate the completely synthetic datasets (SynSparse and SynDense) and the semi-synthetic quick-commerce dataset are present in ./data_gen. Exemplar instances of these datasets are provided in the ./data/synthetic and ./data/quick_commerce respectively. 
