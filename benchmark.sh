#!/bin/bash
# A small script to easily benchmark static vs dynamic scaling

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