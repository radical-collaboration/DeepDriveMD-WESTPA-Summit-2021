#!/bin/bash

if [ -n "$SEG_DEBUG" ] ; then
  set -x
  env | sort
fi

cd $WEST_SIM_ROOT

RMSD=$(mktemp)
COM=$(mktemp)

module load hdf5
python_path=/gpfs/alpine/world-shared/bip216/ddmd_westpa/envs/pytorch/bin/python
python_path=/tmp/pytorch/bin/python
pcoord_file=/tmp/$(uuidgen).txt

${python_path} /tmp/deepdrivemd.py \
  --top /tmp/closed.pdb \
  --coord $WEST_STRUCT_DATA_REF \
  --output_path ${pcoord_file} \
  --ref /tmp/spike_WE.pdb \
  --selection "protein and name CA" \
  --model_cfg /tmp/aae_template.yaml \
  --model_weights /tmp/epoch-100-20210727-180344.pt \
  --batch_size 1 \
  --device cpu \
  --pcoord_dim 2


cat ${pcoord_file}>$WEST_PCOORD_RETURN
rm ${pcoord_file}

if [ -n "$SEG_DEBUG" ] ; then
    head -v $WEST_PCOORD_RETURN
fi

