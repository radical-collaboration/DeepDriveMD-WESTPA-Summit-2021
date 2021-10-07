#!/bin/bash
#BSUB -P BIP216
#BSUB -q debug
#BSUB -J we-init
#BSUB -o init.%J.initout
#BSUB -e init.%J.initerr
#BSUB -nnodes 1
#BSUB -W 00:10

set -x

echo "start init"
./init.sh
echo "finish init"

