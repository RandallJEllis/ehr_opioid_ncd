#!/bin/bash 
#BSUB -P acc_DADisorders
#BSUB -q premium
#BSUB -n 1
#BSUB -R rusage[mem=16000]
#BSUB -W 24:00
#BSUB -o stdout.%J.%I
#BSUB -e stderr.%J.%I

module load python
module load openssl
module load udunits


python3 opioid_voe_controlsNoOpioids.py
