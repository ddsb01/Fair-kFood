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
	* After getting the data, structure it in the format specified in [./data/A/meta.md](./data/A/meta.md)
* The scripts to generate the completely synthetic datasets (SynSparse and SynDense) and the semi-synthetic quick-commerce dataset are present in [./data_gen/Q](./data_gen/Q) and [./data_gen/Q](./data_gen/Q), respectively. Ensure that the datasets follows the format specified in [./data/X/meta.md](./data/X/meta.md) for synthetich dataset(s) and in [./data/Q/meta.md](./data/Q/meta.md) quick-commerce dataset.
	* To get exemplar, already generated, instances of these datasets, please mail [here](damandeepddsb@gmail.com) with "[FAIR-KSERVER]: DATA REQUEST" as the subject. 


## Running the code
* Offline-optimal solution
	* Usage
	```bash
	python3 src/offline_solution.py --city X --day 1 --objective maxmin
	```
	* The parameters are explained below:
		- **city_name**: city name of the corresponding dataset [default 'X' <-- synthetic dataset]
		```
			A --> Food-delivery 
			Q --> Quick-commerce 
			X --> Synthetic 		
		```
		- **day**: Day corresponding to the dataset [default '1']
		```
			{1,2,3,4,5,6} for city A
			{1} for cities X and Q
		```
		- **objective**: Offline solution objective function [default 'maxmin']
		```
			maxmin --> maxmimize the minimum reward (corresponds to FlowMILP)
			bound --> objective with cost-efficiency constraint (corresponds to FlowMILP (2S))
			min --> minimize the net reward
			multi --> 'max' objective with cost-efficiency constraint
		```	
		- for other auxiliary arguments, refer `./src/offline_solution.py`
* Online solutions
	* Usage
	```bash
	python3 src/online_solution.py --city X
	```
	* The parameters are explained below:
		- **city_name**: city name of the corresponding dataset [default 'X' <-- synthetic dataset]
		```
			A --> Food-delivery 
			Q --> Quick-commerce 
			X --> Synthetic 		
		```
		- **day**: Day corresponding to the dataset [default '1']
		```
			{1,2,3,4,5,6} for city A
			{1} for cities X and Q
		```
		- **method**: Online algorithm [default 'doc4food']
		```
			random --> random assignment preferring server with the least reward so far (RANDOM algorithm)
			min --> assigns request to the server with the least reward (GREEDYMIN algorithm)
			min* --> 'min' with considerating of virtual movements (Doc4Food algorithm)
			round-robin --> round-robin assignment of requests (RoundRobiin algorithm)
		```	
		- for other auxiliary arguments, refer `./src/online_solution.py`

