#!/bin/bash 
#BSUB -P acc_DADisorders
#BSUB -q premium
#BSUB -n 32
#BSUB -R span[ptile=3]
#BSUB -R rusage[mem=32000]
#BSUB -W 22:00
#BSUB -o stdout.%J.%I
#BSUB -e stderr.%J.%I

module load R
module load openssl
module load udunits

R CMD BATCH create_matched_cohorts.R
