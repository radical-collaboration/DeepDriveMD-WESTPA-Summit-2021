#!/bin/bash
#SBATCH -J 1-node_128-gbs        # Job name
#SBATCH -o %j.out                # Name of stdout output file
#SBATCH -e %j.err                # Name of stderr error file
#SBATCH -p v100                  # Queue (partition) name
#SBATCH -N 1                     # Total # of nodes (must be 1 for serial)
#SBATCH -n 4                     # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 12:00:00              # Run time (hh:mm:ss)
 
# Other commands must follow all #SBATCH directives ...
module load conda
conda activate /scratch/06079/tg853783/ddmd/envs/pytorch.mpi
export HDF5_USE_FILE_LOCKING='FALSE'

bash_script="/scratch/06079/tg853783/ddmd/src/DeepDriveMD-Longhorn-2021/longhorn.sh"
python_exe="/scratch/06079/tg853783/ddmd/envs/pytorch.mpi/bin/python"
train_script="/scratch/06079/tg853783/ddmd/src/DeepDriveMD-Longhorn-2021/ddp_aae_experiments/train.py"
config_file="/scratch/06079/tg853783/ddmd/src/DeepDriveMD-Longhorn-2021/ddp_aae_experiments/aae_ddp_template.yaml"

# Launch MPI code ...
ibrun -n 4 ${bash_script} ${python_exe} ${train_script} "-c" ${config_file} "--output_path" ${output_path}
