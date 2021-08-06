#!/bin/bash
#SBATCH -p v100
#SBATCH -J final-spike-we
#SBATCH -o final.%j.%N.out
#SBATCH -e final.%j.%N.err
#SBATCH -N 20
#SBATCH -n 80
#SBATCH -t 00:25:00

set -x
cd $SLURM_SUBMIT_DIR
source ~/.profile
export MY_SPECTRUM_OPTIONS="--gpu"
export PATH=$PATH:$HOME/bin
export WEST_ROOT=/scratch/06079/tg853783/ddmd/envs/westpa/westpa-2020.03
source /scratch/06079/tg853783/ddmd/envs/westpa/westpa-2020.03/westpa.sh
alias python='/scratch/06079/tg853783/ddmd/envs/westpa/bin/python'
module load launcher-gpu
module load cuda/10.1
module use /scratch/apps/modulefiles
module load amber/18.0
module load conda
conda activate /scratch/06079/tg853783/ddmd/envs/westpa
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export WEST_SIM_ROOT=$SLURM_SUBMIT_DIR
export PYTHONPATH=/scratch/06079/tg853783/ddmd/envs/westpa/bin/python
export WEST_PYTHON=/scratch/06079/tg853783/ddmd/envs/westpa/bin/python
source env.sh || exit 1
env | sort
SERVER_INFO=$WEST_SIM_ROOT/west_zmq_info-$SLURM_JOBID.json

#TODO: set num_gpu_per_node
num_gpu_per_node=4
#cuda_file=$PBS_O_WORKDIR/cuda_devices.txt
rm -rf nodefilelist.txt
#rm -rf $cuda_file
scontrol show hostname $SLURM_JOB_NODELIST > nodefilelist.txt

top_file=/scratch/06079/tg853783/ddmd/src/DeepDriveMD-Longhorn-2021/westpa_deepdrivemd/CONFIG/closed.prmtop
pdb_file=/scratch/06079/tg853783/ddmd/src/DeepDriveMD-Longhorn-2021/westpa_deepdrivemd/CONFIG/closed.pdb
prod_in=/scratch/06079/tg853783/ddmd/src/DeepDriveMD-Longhorn-2021/westpa_deepdrivemd/CONFIG/prod.in
ref_pdb=/scratch/06079/tg853783/ddmd/data/raw/spike_WE.pdb
model_weights=/scratch/06079/tg853783/ddmd/runs/ddp_aae_experiments/1-node_128-gbs/checkpoint/epoch-100-20210727-180344.pt
model_config=/scratch/06079/tg853783/ddmd/src/DeepDriveMD-Longhorn-2021/ddp_aae_experiments/aae_template.yaml
model_import=/scratch/06079/tg853783/ddmd/src/DeepDriveMD-Longhorn-2021/ddp_aae_experiments/aae_config.py
#runseg=/scratch/06079/tg853783/ddmd/src/DeepDriveMD-Longhorn-2021/westpa_deepdrivemd/runseg.sh
deepdrivemd=/scratch/06079/tg853783/ddmd/src/DeepDriveMD-Longhorn-2021/westpa_deepdrivemd/deepdrivemd.py
seg_id=/scratch/06079/tg853783/ddmd/src/DeepDriveMD-Longhorn-2021/westpa_deepdrivemd/seg_id.py
static_files="${top_file} ${pdb_file} ${prod_in} ${ref_pdb} ${model_weights} ${model_config} ${model_import} ${deepdrivemd} ${seg_id}"

#conda_path=/scratch/06079/tg853783/ddmd/envs/pytorch.mpi
conda_path=/scratch/06079/tg853783/ddmd/envs/pytorch_cloned_on_tmp.tar


# Copy input data to each node's local storage /tmp and make tmp output dir
# Need extra /tmp/traj_segs incase node failed to clean up properly
for i in $(cat nodefilelist.txt)
do
    ssh ${i} "cp ${static_files} /tmp; rm -r /tmp/traj_segs; mkdir /tmp/traj_segs; tar -xf ${conda_path} -C /tmp" & 
done
wait


# start server
$WEST_ROOT/bin/w_run --debug --work-manager=zmq --n-workers=0 --zmq-mode=master --zmq-write-host-info=$SERVER_INFO --zmq-comm-mode=tcp &> west-$SLURM_JOBID-local.log &

# wait on host info file up to 1 min
for ((n=0; n<60; n++)); do
    if [ -e $SERVER_INFO ] ; then
        echo "== server info file $SERVER_INFO =="
        cat $SERVER_INFO
        break
    fi
    sleep 1
done

# exit if host info file doesn't appear in one minute
if ! [ -e $SERVER_INFO ] ; then
    echo 'server failed to start'
    exit 1
fi
export CUDA_VISIBLE_DEVICES=0,1,2,3
for node in $(cat nodefilelist.txt); do
    ssh -o StrictHostKeyChecking=no $node $PWD/node.sh $SLURM_SUBMIT_DIR $SLURM_JOBID $node $CUDA_VISIBLE_DEVICES --work-manager=zmq --n-workers=$num_gpu_per_node --zmq-mode=client --zmq-read-host-info=$SERVER_INFO --zmq-comm-mode=tcp &
done
wait

