#!/bin/bash
#SBATCH --gres=gpu:volta:1
DATE_FOLDER="apr21-c"
mkdir -p "what_did_we_learn/$DATE_FOLDER"

DATASET="hpo_metab"
RESULT_FOLDER="what_did_we_learn/$DATE_FOLDER/$DATASET"
mkdir -p "$RESULT_FOLDER"
echo "Running Graph Mamba on dataset: $DATASET"
echo "Logging results in: $RESULT_FOLDER" 

mkdir -p "$RESULT_FOLDER/1hop_$DATASET"
python learn_graph_mamba_savegrad.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/1hop_$DATASET" --n-layers 1 --epochs 100 --seqlength 1000 --hops 1
mkdir -p "$RESULT_FOLDER/2hop_$DATASET"
python learn_graph_mamba_savegrad.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/2hop_$DATASET" --n-layers 1 --epochs 100 --seqlength 3000 --hops 2
mkdir -p "$RESULT_FOLDER/3hop_$DATASET"
python learn_graph_mamba_savegrad.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/3hop_$DATASET" --n-layers 1 --epochs 100 --seqlength 5000 --hops 3

DATASET="hpo_neuro"
RESULT_FOLDER="what_did_we_learn/$DATE_FOLDER/$DATASET"
mkdir -p "$RESULT_FOLDER"
echo "Running Graph Mamba on dataset: $DATASET"
echo "Logging results in: $RESULT_FOLDER" 

mkdir -p "$RESULT_FOLDER/1hop_$DATASET"
python learn_graph_mamba_savegrad.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/1hop_$DATASET" --n-layers 1 --epochs 100 --seqlength 1000 --hops 1
mkdir -p "$RESULT_FOLDER/2hop_$DATASET"
python learn_graph_mamba_savegrad.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/2hop_$DATASET" --n-layers 1 --epochs 100 --seqlength 3000 --hops 2
mkdir -p "$RESULT_FOLDER/3hop_$DATASET"
python learn_graph_mamba_savegrad.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/3hop_$DATASET" --n-layers 1 --epochs 100 --seqlength 5000 --hops 3

DATASET="ppi_bp"
RESULT_FOLDER="what_did_we_learn/$DATE_FOLDER/$DATASET"
mkdir -p "$RESULT_FOLDER"
echo "Running Graph Mamba on dataset: $DATASET"
echo "Logging results in: $RESULT_FOLDER" 

mkdir -p "$RESULT_FOLDER/1hop_$DATASET"
python learn_graph_mamba_savegrad.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/1hop_$DATASET" --n-layers 1 --epochs 100 --seqlength 1000 --hops 1
mkdir -p "$RESULT_FOLDER/2hop_$DATASET"
python learn_graph_mamba_savegrad.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/2hop_$DATASET" --n-layers 1 --epochs 100 --seqlength 3000 --hops 2
mkdir -p "$RESULT_FOLDER/3hop_$DATASET"
python learn_graph_mamba_savegrad.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/3hop_$DATASET" --n-layers 1 --epochs 100 --seqlength 5000 --hops 3
