#!/bin/bash
#SBATCH --gres=gpu:volta:1

DATASET="hpo_metab"
echo "Running Subgraph Mamba on dataset: $DATASET"
python learn.py --dataset "$DATASET" --logfilename "apr11-d/1layer_$DATASET" --n-layers 1 --concat-emb --graph-conv --zero-one-label 
python learn.py --dataset "$DATASET" --logfilename "apr11-d/2layer_$DATASET" --n-layers 2 --concat-emb --graph-conv --zero-one-label 
python learn.py --dataset "$DATASET" --logfilename "apr11-d/4layer_$DATASET" --n-layers 4 --concat-emb --graph-conv --zero-one-label 

python learn.py --dataset "$DATASET" --logfilename "apr11-d/frozen_$DATASET" --n-layers 1 --concat-emb --graph-conv --zero-one-label --freeze-mamba
python learn.py --dataset "$DATASET" --logfilename "apr11-d/no_01_$DATASET" --n-layers 1 --concat-emb --graph-conv 
python learn.py --dataset "$DATASET" --logfilename "apr11-d/no_concat_$DATASET" --n-layers 1 --graph-conv --zero-one-label 
python learn.py --dataset "$DATASET" --logfilename "apr11-d/no_gcn_$DATASET" --n-layers 1 --concat-emb --zero-one-label 