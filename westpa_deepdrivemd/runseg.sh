#!/bin/bash

if [ -n "$SEG_DEBUG" ] ; then
  set -x
  env | sort
fi

cd $WEST_SIM_ROOT
mkdir -pv $WEST_CURRENT_SEG_DATA_REF
cd $WEST_CURRENT_SEG_DATA_REF

cp $WEST_SIM_ROOT/CONFIG/closed.prmtop .
cp $WEST_SIM_ROOT/analysis.cpptraj .

if [ "$WEST_CURRENT_SEG_INITPOINT_TYPE" = "SEG_INITPOINT_CONTINUES" ]; then
  sed "s/RAND/$WEST_RAND16/g" $WEST_SIM_ROOT/CONFIG/prod.in > prod.in
  cp $WEST_PARENT_DATA_REF/seg.restrt ./parent.restrt
elif [ "$WEST_CURRENT_SEG_INITPOINT_TYPE" = "SEG_INITPOINT_NEWTRAJ" ]; then
  sed "s/RAND/$WEST_RAND16/g" $WEST_SIM_ROOT/CONFIG/prod.in > prod.in
  cp $WEST_PARENT_DATA_REF ./parent.restrt
fi

export CUDA_DEVICES=(`echo $CUDA_VISIBLE_DEVICES_ALLOCATED | tr , ' '`)
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES[$WM_PROCESS_INDEX]}

echo "RUNSEG.SH: CUDA_VISIBLE_DEVICES_ALLOCATED = " $CUDA_VISIBLE_DEVICES_ALLOCATED
echo "RUNSEG.SH: WM_PROCESS_INDEX = " $WM_PROCESS_INDEX
echo "RUNSEG.SH: CUDA_VISIBLE_DEVICES = " $CUDA_VISIBLE_DEVICES

# Runs dynamics
$PMEMD -O -p closed.prmtop    -i   prod.in  -c parent.restrt  -o seg.out           -inf seg.nfo -l seg.log -x seg.nc            -r   seg.restrt || exit 1

# Calculate dynamics
#$CPPTRAJ -i analysis.cpptraj

# Write code in external python script
#def f():
#  take westpa input
#  preprocess
#  pc = call AI in inference
#  write pc

# WEST_PCOORD_RETURN holds the progress coorinates
# If it recieves the wrong shape it will crash
# in 1d case: a single text file, each line containing a floating value
# in 2d case: every line contains two values separated by white space
# each line in these files corresponds to a frame of the trajcetory
# This shape is defined in west.cfg
# Need to replace below line with output of my function
#paste <(cat rbd_comA.dat | tail -n +2 | awk {'print $2'}) <(cat rbd_rmsdA.dat | tail -n +2 | awk {'print $2'})>$WEST_PCOORD_RETURN
#/scratch/06079/tg853783/ddmd/envs/pytorch.mpi/bin/python $WEST_SIM_ROOT/deepdrivemd.py -t $WEST_SIM_ROOT/CONFIG/closed.prmtop -c seg.nc
#python $WEST_SIM_ROOT/deepdrivemd.py
#cat $WEST_SIM_ROOT/pcoord.txt > $WEST_PCOORD_RETURN

python_path=/scratch/06079/tg853783/ddmd/envs/pytorch.mpi/bin/python
pcoord_file=$WEST_SIM_ROOT/PCOORDS/$(uuidgen).txt
ambpdb -p closed.prmtop -c parent.restrt > parent.pdb

${python_path} $WEST_SIM_ROOT/deepdrivemd.py  \
  --top $WEST_SIM_ROOT/CONFIG/closed.pdb \
  --coord seg.nc \
  --output_path ${pcoord_file} \
  --parent parent.pdb \
  --ref /scratch/06079/tg853783/ddmd/data/raw/spike_WE.pdb \
  --selection "protein and name CA" \
  --model_cfg /scratch/06079/tg853783/ddmd/src/DeepDriveMD-Longhorn-2021/ddp_aae_experiments/aae_template.yaml \
  --model_weights /scratch/06079/tg853783/ddmd/runs/ddp_aae_experiments/1-node_128-gbs/checkpoint/epoch-100-20210727-180344.pt \
  --batch_size 32 \
  --device cpu

cat ${pcoord_file}>$WEST_PCOORD_RETURN
rm ${pcoord_file}


#cat rbd_rmsdB.dat | tail -n +2 | awk {'print $2'} > $WEST_RBD_RMSDB_RETURN
#cat rbd_rmsdC.dat | tail -n +2 | awk {'print $2'} > $WEST_RBD_RMSDC_RETURN

#cat rbd_comB.dat | tail -n +2 | awk {'print $2'} > $WEST_RBD_COMB_RETURN
#cat rbd_comC.dat | tail -n +2 | awk {'print $2'} > $WEST_RBD_COMC_RETURN

#cat rbd_angleA.dat | tail -n +2 | awk {'print $2'} > $WEST_RBD_ANGLEA_RETURN
#cat rbd_angleB.dat | tail -n +2 | awk {'print $2'} > $WEST_RBD_ANGLEB_RETURN
#cat rbd_angleC.dat | tail -n +2 | awk {'print $2'} > $WEST_RBD_ANGLEC_RETURN

#cat n165_glycan.dat | tail -n +2 | awk {'print $2'} > $WEST_N165_GLYCAN_RETURN 
#cat n234_glycan.dat | tail -n +2 | awk {'print $2'} > $WEST_N234_GLYCAN_RETURN
#cat n343_glycan.dat | tail -n +2 | awk {'print $2'} > $WEST_N343_GLYCAN_RETURN

mv seg_nosolvent.nc seg.nc
rm -f prod.in closed.prmtop analysis.cpptraj
