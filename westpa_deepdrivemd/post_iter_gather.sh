#!/bin/bash

if [ -n "$SEG_DEBUG" ] ; then
    set -x
    env | sort
fi

# Make iteration output directory
output_dir=$WEST_SIM_ROOT/traj_segs/$(printf "%06d" $WEST_CURRENT_ITER)
mkdir -p ${output_dir}

# Compress simulation outputs on node local storage and move them
# to the file system. Do this for each node in parallel.
for i in $(cat nodefilelist.txt)
do
    ssh ${i} "cd /tmp; tar -czf ${i}.tar.gz traj_segs/*; mv ${i}.tar.gz ${output_dir}; rm -r /tmp/traj_segs/*" &
done
wait

# Once the simulation output files are transfered, untar and write
# to correct ouput directory for WESTPA. tar files contain a traj_segs dir
# which then contains dirs labeled by seg_id, handle this with a ${i} temp dir
cd ${output_dir}
for i in $(cat $WEST_SIM_ROOT/nodefilelist.txt)
do
   mkdir ${output_dir}/${i}
   tar -xzf ${i}.tar.gz -C ${output_dir}/${i}
   mv ${output_dir}/${i}/traj_segs/* ${output_dir}
   rm -r ${output_dir}/${i} ${i}.tar.gz
done

