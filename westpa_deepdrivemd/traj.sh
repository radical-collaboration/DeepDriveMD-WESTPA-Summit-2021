#!/bin/bash


# --------------------------------
# Amber Trajectory Tool for WESTPA
# --------------------------------
# 
# Written by Anthony Bogetti on 28.08.18
# 
# This script will stitch together a trajectory file from your Amber-WESTPA
# simulation that can be viewed in VMD or another molecular dynmaics 
# visualization software.  Run this script with the command ./amberTraj.sh
# from the same directory where the west.h5 file from your WESTPA simulation
# is located.  The results of this analysis will be stored in a new folder
# called trajAnalysis as the file trace.nc.  Load trace.nc into VMD to 
# visualize the trajectory.  As a note, you will need to have your computer
# configured to run w_succ from the WESTPA software package and cpptraj from 
# the Amber software package.  Though, if the simulation has completed successfully,
# these commands will most likely be ready to run.


# The variables defined below are the name of the new analysis directory that
# will be created and the name of an intermediate file in the process of 
# stitching together the trajectory file.
dir=trajAnalysis
#file=path.txt
TOP=/oasis/scratch/comet/tsztainp/temp_project/FORKED_2/CONFIG/closed_strip.prmtop
siter=205
sseg=887
export CPPTRAJ=$AMBERHOME/bin/cpptraj.MPI

# initial state of the system, which doesn't have an iter:seg ID)
cat $(echo 'traj_'$siter'_'$sseg'_trace.txt') | tail -n +9 > path.txt


# users, however, the following should work just fine.

while read file; do
	iter=$(echo $file | awk '{print $1}')
	seg=$(echo $file | awk '{print $2}')
	#filestring='../traj_segs/'$(printf "%06d" $iter)'/'$(printf "%06d" $seg)'/''seg.nc' 
	#tar -zxvf ../traj_segs/.tar traj_segs/000001/000009/seg.nc
	#tar --extract file='../traj_segs/'$(printf "%06d" $iter)'.tar' $filestring 
	tarf='../traj_segs/'$(printf "%06d" $iter)'.tar'
	filestring='traj_segs/'$(printf "%06d" $iter)'/'$(printf "%06d" $seg)'/''seg.nc'
	tar -xvf $tarf $filestring
	echo "trajin $filestring" >> cpptraj.in	
done < "path.txt"

# These two lines will specify the name of the file where the stitched rtajectory
# is written to and a line to commence the cpptraj run
printf "trajout trace.nc\nrun" >> cpptraj.in 

# Now, cpptraj is called using the NaCl parameter file and the cpptraj.in file


