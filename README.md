# DeepDriveMD-Longhorn-2021
Scripts, configs, results, documentation, and analysis of DeepDriveMD experiments run on Longhorn in 2021


Everything related to the experiment is stored here `/scratch/06079/tg853783/ddmd` on Longhorn.

# Data
We benchmark the Adversarial Autoencoder (AAE) model on the Spike protein simulations from WESTPA which can be downloaded here:
https://amarolab.ucsd.edu/files/covid19/TRAJECTORIES_continuous_spike_opening_WE_chong_and_amarolab.tar.gz

We store the data on Longhorn here: `/scratch/06079/tg853783/ddmd/data`

### Download instructions
Make a `data` directory, `cd` into it and follow these steps:
```
curl https://amarolab.ucsd.edu/files/covid19/TRAJECTORIES_continuous_spike_opening_WE_chong_and_amarolab.tar.gz --output TRAJECTORIES_continuous_spike_opening_WE_chong_and_amarolab.tar.gz

tar -xvf TRAJECTORIES_continuous_spike_opening_WE_chong_and_amarolab.tar.gz

rm TRAJECTORIES_continuous_spike_opening_WE_chong_and_amarolab.tar.gz
```

The size of this dataset is:
```
$ du -h *
4.0K	README.txt
1.3G	spike_WE.dcd
26M	spike_WE.prmtop
5.9M	spike_WE_renumbered.psf
```

### Larger dataset
A larger dataset containing 130880 examples can be found here: `/scratch/06079/tg853783/ddmd/data/spike-all.h5`

The size of this dataset is:
```
$ du -h spike-all.h5
23G	spike-all.h5
```


# Environments
The conda environment for running DeepDriveMD and the AAE training in offline mode can be found here: `/scratch/06079/tg853783/ddmd/envs/ddmd`

To load the environment:
```
module load conda
conda activate /scratch/06079/tg853783/ddmd/envs/ddmd
```

To reproduce the creation of this environment please see: `env_recipe.txt`

### Radical EnTK
Environment variables can be configured by running: 
```
source ~/.radical/auth
```
For help with configuring this file, please reach out to our team by posting an issue in the [DeepDriveMD](https://github.com/DeepDriveMD/DeepDriveMD-pipeline) repository.

# Source Code
The source code, including that of [DeepDriveMD](https://github.com/DeepDriveMD/DeepDriveMD-pipeline) and this repository, can be found here: `/scratch/06079/tg853783/ddmd/src/`
