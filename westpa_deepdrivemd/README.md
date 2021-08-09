# WESTPA + DeepDriveMD integration

This implementation uses WESTPA to manage the simulation workflow whose main entry point is `runwe.sh`. 
Each iteration of WESTPA, runs many parrallel instances of `runseg.sh` which runs a segement 
(simulation + AI) within the WESTPA framework. The `runseg.sh` script first runs molecular dynamics and
outputs it's trajectory file. It then passes the new trajectory to the `deepdrivemd.py` script which
implements the preprocessing, and AI inference portions of the DeepDriveMD framework. To be clear, this
script does **NOT** run the DeepDriveMD worfklow as implemented with radical.entk. We are also **NOT** training
the machine learning model here, but instead are using pre-trained model weights taken from the 
[ddp_aae_experiments](https://github.com/DeepDriveMD/DeepDriveMD-Longhorn-2021/blob/main/ddp_aae_experiments/README.md)
results. The `deepdrivemd.py` script is ultimately responsible for inputting the trajectory, and outputting a
`pcoord.txt` file which contains the first two latent dimensions of the adversarial autoencoder model acting as a 
progress coordinate. This file is passed to WESTPA to update it's sampling distribution and spawn the next iteration 
of simulations according to the search space.

In order to run this experiment, WESTPA first needs to be initialized via: `sbatch runinit.sh`. This is a quick
job that should run in under a minute. It is responsible for running the AAE model on each of the initial coordinate
configurations in the `bstates` folder. We have staged all possible files including the pytorch conda environment to 
the node local storage in `/tmp`, please see `init.sh` for this logic. We found it necessary to stage the conda environment
esspecially when there are many parallel workers, otherwise nodes can become unrepsonsive and stall the entire workflow.
The `init.sh` script uses the WESTPA framework to call `get_pcoord.sh` in parallel, for each state in `bstates` (about 50),
this file contains a call to the `deepdrivemd.py` script.

Once WESTPA is initialized, the full workflow can be run via: `sbatch runwe.sh`. It will output results in the `traj_segs`
directory and individual logs from each of the parallel `runseg.sh` calls are routed to `seg_logs` (these are useful to see
if the workflow is running correctly). The main log to watch will output to this directory under the name `west-<jobid>-local.log`.
The current run will use 20 nodes and should finish within 25 minutes. Where possible, files, scripts, and conda environments are
staged to the node local `/tmp` directory. The `deepdrivemd.py` script and input files are completely isolated to use `/tmp` and
does not interact with the parallel file system. We also write all simulation outputs to `/tmp` and then copy it back in batches
using the `post_iter_gather.sh` script at the end of each WESTPA iteration. In this way, each node uses each of it's GPUs to run
one segment (`runseg.sh`) or more depending on if there are more segments to run than available GPU workers, and then waits to transfer
the output simulation data back to the parallel file system until all the work is done for each segment. To further minimize I/O, we
also tar and compress the outputs computed on each node before transfering back to the parallel file system for storage. This way we
only run a single move command for each node to transfer the data back from potentially 100s of parallel simulations.
