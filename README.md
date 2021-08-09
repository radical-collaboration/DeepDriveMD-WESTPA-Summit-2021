# DeepDriveMD-Longhorn-2021
Scripts, configs, results, documentation, and analysis of DeepDriveMD experiments run on Longhorn in 2021

Everything related to the experiment is stored here `/scratch/06079/tg853783/ddmd` on Longhorn with a group permission to read/write by `G-822428`.

Please reach out to our team by posting an issue in the [DeepDriveMD](https://github.com/DeepDriveMD/DeepDriveMD-pipeline) in case you need access to the shared path.

# Data
We benchmark the Adversarial Autoencoder (AAE) model on the spike protein simulations from WESTPA which 
we have collected on Longhorn here: `/scratch/06079/tg853783/ddmd/data/raw`

The data is 94GB in total, this includes 1 PDB file and 32 DCD files.
```
$ du -h /scratch/06079/tg853783/ddmd/data/raw/
94G	/scratch/06079/tg853783/ddmd/data/raw/
```

Please reach out to our team by posting an issue in the [DeepDriveMD](https://github.com/DeepDriveMD/DeepDriveMD-pipeline) repository in case you need access to the original dataset.

The prepreprocessed data can be found here: `/scratch/06079/tg853783/ddmd/data/preprocessed/spike-all-AAE.h5` or recomputed with the preprocessing script (see below).

The dataset contains 130880 frames of MD data which can be used to train the AAE:
```
h5ls /scratch/06079/tg853783/ddmd/data/preprocessed/spike-all-AAE.h5
point_cloud              Dataset {130880, 3, 3375}
rmsd                     Dataset {130880}
```

# Environments
The conda environments for running DeepDriveMD and the AAE training in offline mode can be found here:
* DeepDriveMD: `/scratch/06079/tg853783/ddmd/envs/ddmd`
* PyTorch for AAE training: `/scratch/06079/tg853783/ddmd/envs/pytorch.mpi`

To load the DeepDriveMD environment:
```
module load conda
conda activate /scratch/06079/tg853783/ddmd/envs/ddmd
```

To reproduce the creation of these environments, please see: `env_recipe.txt`

### Environment Variables for RADICAL EnTK
To run the DeepDriveMD workflow, the environment variables can be configured by running: 
```
source ~/.radical/auth
```
For help with configuring this file, please reach out to our team by posting an issue in the [DeepDriveMD](https://github.com/DeepDriveMD/DeepDriveMD-pipeline) repository.

# Source Code
The source code, including that of [DeepDriveMD](https://github.com/DeepDriveMD/DeepDriveMD-pipeline) and this repository, can be found here: `/scratch/06079/tg853783/ddmd/src/`

# Running the Code

### Preprocessing
To preprocess the raw spike protein simulation data in `/scratch/06079/tg853783/ddmd/data/raw`,
run the `preprocess.py` script. This script will copy the raw data to the node's ssd, and then spawn a parallel process 
for each input trajectory file `*.dcd` which aligns each frame of the simulation to a reference structure and then 
collects the raw 3D positions of the CA atoms and the root mean squared deviation (for plotting) of each frame with 
respect to the reference structure. We use the [MDAnalysis](https://www.mdanalysis.org/) package to process the 
trajectories, which also spawns workers. Thus, this code will utilize all cores on a node at near 100%. Follow these steps 
to run the code:
```
idev -m 15 -n 1 -N 1
module load conda
conda activate /scratch/06079/tg853783/ddmd/envs/pytorch.mpi
export HDF5_USE_FILE_LOCKING='FALSE'
python preprocess.py --raw /scratch/06079/tg853783/ddmd/data/raw --preprocessed /scratch/06079/tg853783/ddmd/data/preprocessed --name spike-all-AAE.h5
```

The preprocessing outputs a directory `/scratch/06079/tg853783/ddmd/data/preprocessed` containing
an `h5` file for each input `dcd` file (with the same basename) as well as a concatenated `h5` file
found here: `/scratch/06079/tg853783/ddmd/data/preprocessed/spike-all-AAE.h5`

The preprocessing runtime is reported as follows:
```
Elapsed time: 746.68s
Preprocessing runtime: 320.52s +- 212.24s
```
Note: The mean and standard deviation times report the times taken to process each of the 32 input
`dcd` files. The high standard deviation reflects the fact that some files have many more frames than
others.

***


### Training & Inference

Please see this [README](https://github.com/DeepDriveMD/DeepDriveMD-Longhorn-2021/blob/main/ddp_aae_experiments/README.md).

***

### Outlier Detection

To run outlier detection on the AAE embeddings using scikit-learn's [LocalOutlierFactor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor) method,
run the `outlier_detection.py` script as seen below: 

```
idev -m 10 -n 1 -N 1
module load conda
conda activate /scratch/06079/tg853783/ddmd/envs/pytorch.mpi
cd /scratch/06079/tg853783/ddmd/src/DeepDriveMD-Longhorn-2021

python outlier_detection.py \
--embeddings_path /scratch/06079/tg853783/ddmd/src/DeepDriveMD-Longhorn-2021/ddp_aae_experiments/embeddings/1-node_128-gbs_100-epoch.npy \
--score_output_path /scratch/06079/tg853783/ddmd/src/DeepDriveMD-Longhorn-2021/outliers/1-node_128-gbs_outlier_scores.npy \
--n_jobs -1
```

The outlier detection runtime is reported as follows:
```
Elapsed time: 294.05s
```

Note, `n_jobs=-1` allows scikit-learn to use all available processors.

***

### DeepDriveMD Workflow

To run the DeepDriveMD workflow with a template for a particular system structure, invoke `python` in the following way with the example of Spike protein solvated in a water box. This command replaces sbatch with a slurm job script as RADICAL Cybertools manage the job submission internally.:
```
module load conda
conda activate /scratch/06079/tg853783/ddmd/envs/ddmd
python -m deepdrivemd.deepdrivemd --config /scratch/06079/tg853783/ddmd/src/DeepDriveMD-Longhorn-2021/template/spike_waterbox_template.yaml
```

Note, `experiment_directory` in the YAML template MUST not exist prior invoking the DeepDriveMD workflow, it will create automatically when it's started. This is where all outputs are stored from each run of the workflow including OpenMM simulation, preprocessing, training and outlier detection.

***

### WESTPA + DeepDriveMD integration

Please see this [README](https://github.com/DeepDriveMD/DeepDriveMD-Longhorn-2021/blob/main/westpa_deepdrivemd/README.md).

***
