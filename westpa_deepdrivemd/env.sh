# This file defines where WEST and GROMACS can be found
# Modify to taste
source ~/.bash_profile


export MY_SPECTRUM_OPTIONS="--gpu"
export PATH=$PATH:$HOME/bin
export WEST_ROOT=/gpfs/alpine/world-shared/bip216/ddmd_westpa/envs/westpa/westpa-2020.05/
source /gpfs/alpine/world-shared/bip216/ddmd_westpa/envs/westpa/westpa-2020.05/westpa.sh
alias python='/gpfs/alpine/world-shared/bip216/ddmd_westpa/envs/westpa/bin/python'
module load cuda/10.1.243
#module use /scratch/apps/modulefiles
#module load amber/18.0
module load gcc/7.5.0 #mvapich2-gdr/2.3.4 #amber/20.0
#module load conda
. "/sw/summit/python/3.7/anaconda3/5.3.0/etc/profile.d/conda.sh"
conda activate /gpfs/alpine/world-shared/bip216/ddmd_westpa/envs/westpa

source /gpfs/alpine/scratch/hrlee/bip216/amber20/amber.sh

#conda activate westpa-2020.02
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH
# export PATH=$(echo $PATH | sed -e 's|///home/07392/tsztainp/:||g')
export PYTHONPATH=/gpfs/alpine/world-shared/bip216/ddmd_westpa/envs/westpa/bin/python
export WEST_PYTHON=/gpfs/alpine/world-shared/bip216/ddmd_westpa/envs/westpa/bin/python
# Inform WEST where to find Python and our other scripts where to find WEST
if [[ -z "$WEST_ROOT" ]]; then
    echo "Must set environ variable WEST_ROOT"
    exit
fi
# Explicitly name our simulation root directory
if [[ -z "$WEST_SIM_ROOT" ]]; then
    export WEST_SIM_ROOT="$PWD"
fi
export SIM_NAME=$(basename $WEST_SIM_ROOT)
export WEST_ROOT=$WEST_ROOT
echo "simulation $SIM_NAME root is $WEST_SIM_ROOT"

# export NODELOC=/scratch/07392/tsztainp
export USE_LOCAL_SCRATCH=1

export WM_ZMQ_MASTER_HEARTBEAT=100
export WM_ZMQ_WORKER_HEARTBEAT=100
export WM_ZMQ_TIMEOUT_FACTOR=3000
export BASH=$SWROOT/bin/bash
export PERL=$SWROOT/usr/bin/perl
export ZSH=$SWROOT/bin/zsh
export IFCONFIG=$SWROOT/bin/ifconfig
export CUT=$SWROOT/usr/bin/cut
export TR=$SWROOT/usr/bin/tr
export LN=$SWROOT/bin/ln
export CP=$SWROOT/bin/cp
export RM=$SWROOT/bin/rm
export SED=$SWROOT/bin/sed
export CAT=$SWROOT/bin/cat
export HEAD=$SWROOT/bin/head
export TAR=$SWROOT/bin/tar
export AWK=$SWROOT/usr/bin/awk
export PASTE=$SWROOT/usr/bin/paste
export GREP=$SWROOT/bin/grep
export SORT=$SWROOT/usr/bin/sort
export UNIQ=$SWROOT/usr/bin/uniq
export HEAD=$SWROOT/usr/bin/head
export MKDIR=$SWROOT/bin/mkdir
export ECHO=$SWROOT/bin/echo
export DATE=$SWROOT/bin/date
export SANDER=$AMBERHOME/bin/sander
export PMEMD=$AMBERHOME/bin/pmemd.cuda
export CPPTRAJ=$AMBERHOME/bin/cpptraj
