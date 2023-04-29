#!/bin/bash 
#BSUB -P acc_DADisorders
#BSUB -q premium
#BSUB -n 32
#BSUB -R span[ptile=3]
#BSUB -R rusage[mem=16000]
#BSUB -W 22:00
#BSUB -o stdout.%J.%I
#BSUB -e stderr.%J.%I

module load python
module load openssl
module load udunits


python3 voe_prospective_matching_population_preparation.py
