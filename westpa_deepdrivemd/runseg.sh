#!/bin/bash

if [ -n "$SEG_DEBUG" ] ; then
  set -x
  env | sort
fi

echo /tmp/$WEST_CURRENT_SEG_DATA_REF
mkdir -pv /tmp/$WEST_CURRENT_SEG_DATA_REF
cd /tmp/$WEST_CURRENT_SEG_DATA_REF

if [ "$WEST_CURRENT_SEG_INITPOINT_TYPE" = "SEG_INITPOINT_CONTINUES" ]; then
  sed "s/RAND/$WEST_RAND16/g" /tmp/prod.in > ./prod.in
  cp $WEST_PARENT_DATA_REF/seg.restrt ./parent.restrt
elif [ "$WEST_CURRENT_SEG_INITPOINT_TYPE" = "SEG_INITPOINT_NEWTRAJ" ]; then
  sed "s/RAND/$WEST_RAND16/g" /tmp/prod.in > ./prod.in
  cp $WEST_PARENT_DATA_REF ./parent.restrt
fi

export CUDA_DEVICES=(`echo $CUDA_VISIBLE_DEVICES_ALLOCATED | tr , ' '`)
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES[$WM_PROCESS_INDEX]}

echo "RUNSEG.SH: CUDA_VISIBLE_DEVICES_ALLOCATED = " $CUDA_VISIBLE_DEVICES_ALLOCATED
echo "RUNSEG.SH: WM_PROCESS_INDEX = " $WM_PROCESS_INDEX
echo "RUNSEG.SH: CUDA_VISIBLE_DEVICES = " $CUDA_VISIBLE_DEVICES

pwd
ls

# Runs dynamics
$PMEMD -O -p /tmp/closed.prmtop -i ./prod.in -c ./parent.restrt -o ./seg.out -inf ./seg.nfo -l ./seg.log -x ./seg.nc -r ./seg.restrt || exit 1

echo "Finished dynamics, running analysis ..."
ls 

# WEST_PCOORD_RETURN holds the progress coorinates
# If it recieves the wrong shape it will crash
# in 1d case: a single text file, each line containing a floating value
# in 2d case: every line contains two values separated by white space
# each line in these files corresponds to a frame of the trajcetory
# This shape is defined in west.cfg

python_path=/scratch/06079/tg853783/ddmd/envs/pytorch.mpi/bin/python
#pcoord_file=$(uuidgen).txt
# No longer need uuid since each output directory is unique
#pcoord_file=pcoord.txt
ambpdb -p /tmp/closed.prmtop -c ./parent.restrt > ./parent.pdb

ls

${python_path} $WEST_SIM_ROOT/deepdrivemd.py  \
  --top /tmp/closed.pdb \
  --coord ./seg.nc \
  --output_path ./pcoord.txt \
  --parent ./parent.pdb \
  --ref /tmp/spike_WE.pdb \
  --selection "protein and name CA" \
  --model_cfg /tmp/aae_template.yaml \
  --model_weights /tmp/epoch-100-20210727-180344.pt \
  --batch_size 32 \
  --device cuda \
  --pcoord_dim 2

echo "Finished analysis, cleaning up ..."

cat ./pcoord.txt > $WEST_PCOORD_RETURN

#mv seg_nosolvent.nc seg.nc
#rm -f prod.in closed.prmtop closed.pdb spike_WE.pdb aae_template.yaml epoch-100-20210727-180344.pt #analysis.cpptraj

# Remove temporary files
rm ./prod.in ./pcoord.txt
# Move data from node local to file system
mkdir -pv $WEST_CURRENT_SEG_DATA_REF
mv * $WEST_CURRENT_SEG_DATA_REF

