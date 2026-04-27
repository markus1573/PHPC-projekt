#!/bin/bash
### General options 
### -- specify queue -- 
#BSUB -q gpua100
### -- set the job Name -- 
#BSUB -J PHPC_benchmark
### -- ask for number of cores (default: 1) -- 
#BSUB -n 16
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=4GB]"
### -- specify the model of CPU we want to run on --
##BSUB -R "select[model==XeonGold6326]"
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm -- 
#BSUB -W 02:00 
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output_%J.out 
#BSUB -e Output_%J.err 



source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

N=4570
LOAD_DIR="/dtu/projects/02613_2025/data/modified_swiss_dwellings/"

python simulate_CUDA.py $N $LOAD_DIR > full_results.csv
