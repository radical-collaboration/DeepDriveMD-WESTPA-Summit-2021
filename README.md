# DeepDriveMD-Longhorn-2021
Scripts, configs, results, documentation, and analysis of DeepDriveMD experiments run on Longhorn in 2021


Everything related to the experiment is stored here `/scratch/06079/tg853783/ddmd` on Longhorn.

# Data
We benchmark the Adversarial Autoencoder (AAE) model on the Spike protein simulations from WESTPA which can be downloaded here:
https://amarolab.ucsd.edu/files/covid19/TRAJECTORIES_continuous_spike_opening_WE_chong_and_amarolab.tar.gz

We store the data on Longhorn here: `/scratch/06079/tg853783/ddmd/data`

# Environments
The conda environment for running DeepDriveMD and the AAE training in offline mode can be found here: `/scratch/06079/tg853783/ddmd/envs/ddmd`

To load the environment:
```
module load conda
conda activate /scratch/06079/tg853783/ddmd/envs/ddmd
```

To reproduce the creation of this environment please see: TODO

# Source Code
The source code, including that of [DeepDriveMD](https://github.com/DeepDriveMD/DeepDriveMD-pipeline) and this repository, can be found here: `/scratch/06079/tg853783/ddmd/src/`
