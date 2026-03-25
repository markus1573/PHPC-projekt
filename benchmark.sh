#!/bin/bash
# A small script to easily benchmark static vs dynamic scaling

N=20
MAX_CORES=$(python -c "import os; print(os.cpu_count() or 4)")

echo "Benchmarking with N=$N floorplans"
echo "================================================"
echo ""
echo "baseline"
echo "------------------------------------------------"
python simulate_OG.py $N | grep "Process finished"

echo ""
echo "Static Scheduling"
echo "------------------------------------------------"
for w in 2 4 8 $MAX_CORES; do
    echo -n "Workers: $w -> "
    python simulate_static.py $N $w | grep "Process finished"
done

echo ""
echo "Dynamic Scheduling"
echo "------------------------------------------------"
for w in 2 4 8 $MAX_CORES; do
    echo -n "Workers: $w -> "
    python simulate_dynamic.py $N $w | grep "Process finished"
done