#!/bin/bash
#SBATCH -p v100
#SBATCH -J final-spike-we
#SBATCH -o final.%j.%N.out
#SBATCH -e final.%j.%N.err
#SBATCH -N 25
#SBATCH -n 100
#SBATCH -t 01:00:00

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

# start server
$WEST_ROOT/bin/w_run --work-manager=zmq --n-workers=0 --zmq-mode=master --zmq-write-host-info=$SERVER_INFO --zmq-comm-mode=tcp &> west-$SLURM_JOBID-local.log &

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

