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
Additionally, access to Gurobi Optimizer is required. Access to Gurobi's academic (free!) license can be requested [here](https://www.gurobi.com/academia/academic-program-and-licenses/).

## Datasets
* To get access to the food-delivery dataset, please follow the procedure defined [here](https://www.cse.iitd.ac.in/~sayan/files/foodmatch.txt).
	* After getting the data, structure it in the format specified in [./data/A](./data/A)
* The scripts to generate the completely synthetic datasets (SynSparse and SynDense) and the semi-synthetic quick-commerce dataset are present in [./data_gen](./data_gen). 
	* To get exemplar, already generated, instances of these datasets, please mail [here](damandeepddsb@gmail.com) with "[FAIR-KSERVER]: DATA REQUEST" as the subject. 

## Running the code
* Offline-optimal solution
	* Usage
	```bash
	python3 src/offline_solution.py --city X
	```
	* The parameters are explained below:
		- **city_name**: A (food-delivery), Q (quick-commerce), X (synthetic)
		- ****:

* Online solutions
	* Usage
	```bash
	python3 src/online_solution.py --city X
	```
	* The parameters are explained below:
		- **city_name**: A (food-delivery), Q (quick-commerce), X (synthetic)
		- ****:

