# HyperNEAT substrate configuration

## Background

This university project explores the HyperNEAT algorithm and substrate configuration for a directed locomotion task. The motivation is to find the link between subtrate dimensionality and performance and test automation of the substrate configuration process through dimensionality reduction techniques.

## Main Dependencies

The project uses the TensorNEAT implementation of HyperNEAT and the Brax reinforcement learning environment. Weights & Biases (wandb) is used for logging. For data operations and visualizations this project uses matplotlib, pandas, networkx and seaborn.

## Project Structure

### Main Scripts

The main scripts run in a few different notebooks:

* `data_gen.ipynb` is run first to generate data sources from both a random action policy and trained agents.
* `substrate_gen.ipynb` is used to configure substrates bases on the previously generated data sources.
* `neuroevolution.ipynb` runs the HyperNEAT algorithm and to train neural networks for the reinforcement learning environment.
* `evaluation.ipynb` plots some charts to evaluate the results.
* Optional addition: `random_substrate_runs.ipynb` was used to create and evaluate more randomly configured substrates for a greater statistical sample size.

### Config

A central `config.py` is used to set parameters for the scripts.

### Classes and Helper Functions

* All classes related to data sampling and substrate configuration are located in subfolder `substrate_generation`. 
* The parts related to the reinforcement learning and the algorithm itself are located in the subfolder `evol_pipeline`. 
* Additional helper functions for file operations and visualizations can be found in `utils`.

