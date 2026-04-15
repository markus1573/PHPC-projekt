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
#BSUB -R "rusage[mem=1GB]"
### -- specify the model of CPU we want to run on --
#BSUB -R "select[model==XeonGold6126]"
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot -- 
#BSUB -M 5GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 00:10 
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

N=20
LOAD_DIR="subset_swiss_dwellings/"
MAX_CORES=$(python -c "import os; print(os.cpu_count() or 4)")

echo "Benchmarking with N=$N floorplans from ${LOAD_DIR}"
echo "================================================"
echo ""
echo "baseline"
echo "------------------------------------------------"
python simulate_OG.py $N $LOAD_DIR | grep "Process finished"

echo ""
echo "Static Scheduling"
echo "------------------------------------------------"
for w in 2 4 8 $MAX_CORES; do
    echo -n "Workers: $w -> "
    python simulate_static.py $N $w $LOAD_DIR | grep "Process finished"
done

echo ""
echo "Dynamic Scheduling"
echo "------------------------------------------------"
for w in 2 4 8 $MAX_CORES; do
    echo -n "Workers: $w -> "
    python simulate_dynamic.py $N $w $LOAD_DIR | grep "Process finished"
done

echo ""
echo "JIT Simulation"
echo "------------------------------------------------"
python simulate_JIT.py $N $LOAD_DIR | grep "Process finished"

echo ""
echo "CUDA Simulation"
echo "------------------------------------------------"
python simulate_CUDA.py $N $LOAD_DIR | grep "Process finished"

echo ""
echo "cupy Simulation"
echo "------------------------------------------------"
python simulate_cupy.py $N $LOAD_DIR | grep "Process finished"