# SciML_PJT_Implementation
This is a github repo for SciML course project implementation assignment.

This repository contains code implementation of two experiments from the paper:
> *Physics-Informed Neural Networks for Solving Forward and Inverse Problems in Complex Beam System*


## Two Experiments conducted
1. **Noisy vs Clean Observation**
   - 'Main_CleanvsNoisy.py'
   - Compares the robustness and prediction accuracy of the model under noisy measurement data.
   - As the training data points are artifically created based on the anlaytical solution, add of Gaussian noise was implemented to imitate the real world measurement error.
  
2. **Soft vs Hard Boundary Constraint**
    - 'Main_HardvsSoft.py'
    - Implements a hard constraint instead of soft constraint (the paper deploys soft contraint method), and compare predicition accuracy between them, and plots graph of each prediction model at the boundary.
  

## Results
Results will be stored in results/

## Notebook
Jupyter Notebook version of two experiments will be stored in notebooks/
