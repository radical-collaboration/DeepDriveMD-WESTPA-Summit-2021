#!/bin/bash
#BSUB -P BIP216
#BSUB -q batch
#BSUB -J final-spike-we
#BSUB -o final.%J.out
#BSUB -e final.%J.err
#BSUB -nnodes 255
#BSUB -alloc_flags gpumps
#BSUB -alloc_flags smt4
#BSUB -W 01:10

set -x
#export LS_SUBCWD=/gpfs/alpine/world-shared/bip216/ddmd_westpa/src/DeepDriveMD-WESTPA-Summit-2021/westpa_deepdrivemd
cd $LS_SUBCWD
source ~/.bash_profile
export MY_SPECTRUM_OPTIONS="--gpu"
export PATH=$PATH:$HOME/bin
export WEST_ROOT=/gpfs/alpine/world-shared/bip216/ddmd_westpa/envs/westpa/westpa-2020.05/
source /gpfs/alpine/world-shared/bip216/ddmd_westpa/envs/westpa/westpa-2020.05/westpa.sh
alias python='/gpfs/alpine/world-shared/bip216/ddmd_westpa/envs/westpa/bin/python'
#module load launcher-gpu
module load cuda/10.1.243
#module use /scratch/apps/modulefiles
#module load amber/18.0
#module load conda
. "/sw/summit/python/3.7/anaconda3/5.3.0/etc/profile.d/conda.sh"
conda activate /gpfs/alpine/world-shared/bip216/ddmd_westpa/envs/westpa
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export WEST_SIM_ROOT=$LS_SUBCWD
export PYTHONPATH=/gpfs/alpine/world-shared/bip216/ddmd_westpa/envs/westpa/bin/python
export WEST_PYTHON=/gpfs/alpine/world-shared/bip216/ddmd_westpa/envs/westpa/bin/python
source env.sh || exit 1
env | sort
SERVER_INFO=$WEST_SIM_ROOT/west_zmq_info-$LSB_JOBID.json

#TODO: set num_gpu_per_node
num_gpu_per_node=6
#cuda_file=$PBS_O_WORKDIR/cuda_devices.txt
rm -rf nodefilelist.txt
#rm -rf $cuda_file
#scontrol show hostname $SLURM_JOB_NODELIST > nodefilelist.txt
$(cat ${LSB_DJOB_HOSTFILE} | uniq | sort | grep -v batch > nodefilelist.txt)

top_file=/gpfs/alpine/world-shared/bip216/ddmd_westpa/src/DeepDriveMD-WESTPA-Summit-2021/westpa_deepdrivemd/CONFIG/closed.prmtop
pdb_file=/gpfs/alpine/world-shared/bip216/ddmd_westpa/src/DeepDriveMD-WESTPA-Summit-2021/westpa_deepdrivemd/CONFIG/closed.pdb
prod_in=/gpfs/alpine/world-shared/bip216/ddmd_westpa/src/DeepDriveMD-WESTPA-Summit-2021/westpa_deepdrivemd/CONFIG/prod.in
ref_pdb=/gpfs/alpine/world-shared/bip216/ddmd_westpa/data/raw/spike_WE.pdb
model_weights=/gpfs/alpine/world-shared/bip216/ddmd_westpa/data/epoch-100-20210727-180344.pt
model_config=/gpfs/alpine/world-shared/bip216/ddmd_westpa/src/DeepDriveMD-WESTPA-Summit-2021/ddp_aae_experiments/aae_template.yaml
model_import=/gpfs/alpine/world-shared/bip216/ddmd_westpa/src/DeepDriveMD-WESTPA-Summit-2021/ddp_aae_experiments/aae_config.py
#runseg=/gpfs/alpine/world-shared/bip216/ddmd_westpa/src/DeepDriveMD-WESTPA-Summit-2021/westpa_deepdrivemd/runseg.sh
deepdrivemd=/gpfs/alpine/world-shared/bip216/ddmd_westpa/src/DeepDriveMD-WESTPA-Summit-2021/westpa_deepdrivemd/deepdrivemd.py
seg_id=/gpfs/alpine/world-shared/bip216/ddmd_westpa/src/DeepDriveMD-WESTPA-Summit-2021/westpa_deepdrivemd/seg_id.py
static_files="${top_file} ${pdb_file} ${prod_in} ${ref_pdb} ${model_weights} ${model_config} ${model_import} ${deepdrivemd} ${seg_id}"

#conda_path=/gpfs/alpine/world-shared/bip216/ddmd_westpa/envs/pytorch.mpi
conda_path=/gpfs/alpine/world-shared/bip216/ddmd_westpa/envs/pytorch_cloned_on_tmp.tar


# Copy input data to each node's local storage /tmp and make tmp output dir
# Need extra /tmp/traj_segs incase node failed to clean up properly
for i in $(cat nodefilelist.txt)
do
    ssh ${i} "cp ${static_files} /tmp; rm -r /tmp/traj_segs; mkdir /tmp/traj_segs; tar -xf ${conda_path} -C /tmp" & 
done
wait


# start server
$WEST_ROOT/bin/w_run --work-manager=zmq --n-workers=0 --zmq-mode=master --zmq-write-host-info=$SERVER_INFO --zmq-comm-mode=tcp &> west-$LSB_JOBID-local.log &

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
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
for node in $(cat nodefilelist.txt); do
    ssh -o StrictHostKeyChecking=no $node $PWD/node.sh $LS_SUBCWD $LSB_JOBID $node $CUDA_VISIBLE_DEVICES --work-manager=zmq --n-workers=$num_gpu_per_node --zmq-mode=client --zmq-read-host-info=$SERVER_INFO --zmq-comm-mode=tcp &
done
wait

