#!/bin/bash
#SBATCH --gres=gpu:volta:1

DATASET="hpo_metab"
echo "Trying Subgraph Mamba on dataset: $DATASET"
echo "Trying augmenting data by 5x"
python try_things.py --dataset "$DATASET" --augment 5