#!/bin/bash 

source env.sh

SFX=.d$$
mv traj_segs{,$SFX}
mv seg_logs{,$SFX}
mv istates{,$SFX}
rm -Rf traj_segs$SFX seg_logs$SFX istates$SFX & disown %1
rm -f system.h5 west.h5 seg_logs.tar 
rm -rf PCOORDS/* traj_segs seg_logs istates west.h5 west*.log *.json *.txt *.out *.err
mkdir seg_logs traj_segs istates

BSTATE_ARGS="--bstate-file $WEST_SIM_ROOT/BASIS_STATES"
TSTATE_ARGS="--tstate-file $WEST_SIM_ROOT/TSTATE"

# $WEST_ROOT/bin/w_init $BSTATE_ARGS $TSTATE_ARGS --segs-per-state 5 --work-manager=threads "$@"

# Stage files on node local storage
pdb_file=/scratch/06079/tg853783/ddmd/src/DeepDriveMD-Longhorn-2021/westpa_deepdrivemd/CONFIG/closed.pdb
ref_pdb=/scratch/06079/tg853783/ddmd/data/raw/spike_WE.pdb
model_weights=/scratch/06079/tg853783/ddmd/runs/ddp_aae_experiments/1-node_128-gbs/checkpoint/epoch-100-20210727-180344.pt
model_config=/scratch/06079/tg853783/ddmd/src/DeepDriveMD-Longhorn-2021/ddp_aae_experiments/aae_template.yaml
static_files="${pdb_file} ${ref_pdb} ${model_weights} ${model_config}"
echo ${static_files}
cp ${static_files} /tmp

# Remove recycling
$WEST_ROOT/bin/w_init $BSTATE_ARGS --segs-per-state 5 --work-manager=threads "$@"
