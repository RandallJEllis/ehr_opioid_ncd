#!/bin/bash 
#BSUB -P acc_DADisorders
#BSUB -q premium
#BSUB -n 1
#BSUB -R rusage[mem=160000]
#BSUB -W 14:00
#BSUB -o stdout.%J.%I
#BSUB -e stderr.%J.%I

module load R
module load openssl
module load udunits


R CMD BATCH add_dates.R
