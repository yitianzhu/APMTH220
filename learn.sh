#!/bin/bash
#SBATCH --gres=gpu:volta:1
DATE_FOLDER="apr20-c"
mkdir -p "what_did_we_learn/$DATE_FOLDER"

DATASET="hpo_metab"
RESULT_FOLDER="what_did_we_learn/$DATE_FOLDER/$DATASET"
mkdir -p "$RESULT_FOLDER"
echo "Running Subgraph Mamba on dataset: $DATASET"
echo "Logging results in: $RESULT_FOLDER" 

mkdir -p "$RESULT_FOLDER/1layer_$DATASET"
python learn.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/1layer_$DATASET" --n-layers 1 --aggregation "add" --graph-conv --zero-one-label 
mkdir -p "$RESULT_FOLDER/2layer_$DATASET"
python learn.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/2layer_$DATASET" --n-layers 2 --aggregation "add" --graph-conv --zero-one-label 
mkdir -p "$RESULT_FOLDER/4layer_$DATASET"
python learn.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/4layer_$DATASET" --n-layers 4 --aggregation "add" --graph-conv --zero-one-label 

mkdir -p "$RESULT_FOLDER/concat_$DATASET"
python learn.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/concat_$DATASET" --n-layers 1 --aggregation "concat" --graph-conv --zero-one-label 
mkdir -p "$RESULT_FOLDER/frozen_$DATASET"
python learn.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/frozen_$DATASET" --n-layers 1 --aggregation "add" --graph-conv --zero-one-label --freeze-mamba
mkdir -p "$RESULT_FOLDER/no_01_$DATASET"
python learn.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/no_01_$DATASET" --n-layers 1 --aggregation "add" --graph-conv 
mkdir -p "$RESULT_FOLDER/no_skip_$DATASET"
python learn.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/no_skip_$DATASET" --n-layers 1 --graph-conv --zero-one-label 
mkdir -p "$RESULT_FOLDER/no_gcn_$DATASET"
python learn.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/no_gcn_$DATASET" --n-layers 1 --aggregation "add" --zero-one-label 



DATASET="hpo_neuro"
RESULT_FOLDER="what_did_we_learn/$DATE_FOLDER/$DATASET"
mkdir -p "$RESULT_FOLDER"
echo "Running Subgraph Mamba on dataset: $DATASET"
echo "Logging results in: $RESULT_FOLDER" 

mkdir -p "$RESULT_FOLDER/1layer_$DATASET"
python learn.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/1layer_$DATASET" --n-layers 1 --aggregation "add" --graph-conv --zero-one-label 
mkdir -p "$RESULT_FOLDER/2layer_$DATASET"
python learn.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/2layer_$DATASET" --n-layers 2 --aggregation "add" --graph-conv --zero-one-label 
mkdir -p "$RESULT_FOLDER/4layer_$DATASET"
python learn.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/4layer_$DATASET" --n-layers 4 --aggregation "add" --graph-conv --zero-one-label 

mkdir -p "$RESULT_FOLDER/concat_$DATASET"
python learn.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/concat_$DATASET" --n-layers 1 --aggregation "concat" --graph-conv --zero-one-label 
mkdir -p "$RESULT_FOLDER/frozen_$DATASET"
python learn.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/frozen_$DATASET" --n-layers 1 --aggregation "add" --graph-conv --zero-one-label --freeze-mamba
mkdir -p "$RESULT_FOLDER/no_01_$DATASET"
python learn.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/no_01_$DATASET" --n-layers 1 --aggregation "add" --graph-conv 
mkdir -p "$RESULT_FOLDER/no_skip_$DATASET"
python learn.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/no_skip_$DATASET" --n-layers 1 --graph-conv --zero-one-label 
mkdir -p "$RESULT_FOLDER/no_gcn_$DATASET"
python learn.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/no_gcn_$DATASET" --n-layers 1 --aggregation "add" --zero-one-label 




DATASET="ppi_bp"
RESULT_FOLDER="what_did_we_learn/$DATE_FOLDER/$DATASET"
mkdir -p "$RESULT_FOLDER"
echo "Running Subgraph Mamba on dataset: $DATASET"
echo "Logging results in: $RESULT_FOLDER" 

mkdir -p "$RESULT_FOLDER/1layer_$DATASET"
python learn.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/1layer_$DATASET" --n-layers 1 --aggregation "add" --graph-conv --zero-one-label 
mkdir -p "$RESULT_FOLDER/2layer_$DATASET"
python learn.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/2layer_$DATASET" --n-layers 2 --aggregation "add" --graph-conv --zero-one-label 
mkdir -p "$RESULT_FOLDER/4layer_$DATASET"
python learn.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/4layer_$DATASET" --n-layers 4 --aggregation "add" --graph-conv --zero-one-label 

mkdir -p "$RESULT_FOLDER/concat_$DATASET"
python learn.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/concat_$DATASET" --n-layers 1 --aggregation "concat" --graph-conv --zero-one-label 
mkdir -p "$RESULT_FOLDER/frozen_$DATASET"
python learn.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/frozen_$DATASET" --n-layers 1 --aggregation "add" --graph-conv --zero-one-label --freeze-mamba
mkdir -p "$RESULT_FOLDER/no_01_$DATASET"
python learn.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/no_01_$DATASET" --n-layers 1 --aggregation "add" --graph-conv 
mkdir -p "$RESULT_FOLDER/no_skip_$DATASET"
python learn.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/no_skip_$DATASET" --n-layers 1 --graph-conv --zero-one-label 
mkdir -p "$RESULT_FOLDER/no_gcn_$DATASET"
python learn.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/no_gcn_$DATASET" --n-layers 1 --aggregation "add" --zero-one-label 