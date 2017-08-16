#!/bin/bash
#$ -N decision-tree
#$ -q class64-amd 
#$ -pe openmp 64

# Module load gcc compiler version 4.9.2
module load  gcc/4.9.2

# Runs a bunch of standard command-line
# utilities, just as an example:

echo "Script began:" `date`
echo "Node:" `hostname`
echo "Current directory: ${PWD}"

./openmp_DecisionTree.o 200000 784

echo ""
echo "=== Done! ==="

# eof
