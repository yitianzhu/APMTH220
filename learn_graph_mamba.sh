#!/bin/bash
#SBATCH --gres=gpu:volta:1
DATE_FOLDER="apr21-b"
mkdir -p "what_did_we_learn/$DATE_FOLDER"

DATASET="hpo_metab"
RESULT_FOLDER="what_did_we_learn/$DATE_FOLDER/$DATASET"
mkdir -p "$RESULT_FOLDER"
echo "Running Graph Mamba on dataset: $DATASET"
echo "Logging results in: $RESULT_FOLDER" 

mkdir -p "$RESULT_FOLDER/1layer_$DATASET"
python learn_graph_mamba_savegrad.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/1layer_$DATASET" --n-layers 1 --epochs 100
mkdir -p "$RESULT_FOLDER/2layer_$DATASET"
python learn_graph_mamba_savegrad.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/2layer_$DATASET" --n-layers 2 --epochs 100
mkdir -p "$RESULT_FOLDER/4layer_$DATASET"
python learn_graph_mamba_savegrad.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/4layer_$DATASET" --n-layers 4 --epochs 100


DATASET="hpo_neuro"
RESULT_FOLDER="what_did_we_learn/$DATE_FOLDER/$DATASET"
mkdir -p "$RESULT_FOLDER"
echo "Running Graph Mamba on dataset: $DATASET"
echo "Logging results in: $RESULT_FOLDER" 

mkdir -p "$RESULT_FOLDER/1layer_$DATASET"
python learn_graph_mamba_savegrad.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/1layer_$DATASET" --n-layers 1 --epochs 100
mkdir -p "$RESULT_FOLDER/2layer_$DATASET"
python learn_graph_mamba_savegrad.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/2layer_$DATASET" --n-layers 2 --epochs 100
mkdir -p "$RESULT_FOLDER/4layer_$DATASET"
python learn_graph_mamba_savegrad.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/4layer_$DATASET" --n-layers 4 --epochs 100


DATASET="ppi_bp"
RESULT_FOLDER="what_did_we_learn/$DATE_FOLDER/$DATASET"
mkdir -p "$RESULT_FOLDER"
echo "Running Graph Mamba on dataset: $DATASET"
echo "Logging results in: $RESULT_FOLDER" 

mkdir -p "$RESULT_FOLDER/1layer_$DATASET"
python learn_graph_mamba_savegrad.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/1layer_$DATASET" --n-layers 1 --epochs 100
mkdir -p "$RESULT_FOLDER/2layer_$DATASET"
python learn_graph_mamba_savegrad.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/2layer_$DATASET" --n-layers 2 --epochs 100
mkdir -p "$RESULT_FOLDER/4layer_$DATASET"
python learn_graph_mamba_savegrad.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/4layer_$DATASET" --n-layers 4 --epochs 100
