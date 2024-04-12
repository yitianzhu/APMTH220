#!/bin/bash
#SBATCH --gres=gpu:volta:1

DATASET="hpo_metab"
echo "Running Subgraph Mamba on dataset: $DATASET"
python learn.py --dataset "$DATASET"
