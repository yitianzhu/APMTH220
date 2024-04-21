#!/bin/bash
#SBATCH --gres=gpu:volta:1
DATE_FOLDER="apr21-e"
mkdir -p "what_did_we_learn/$DATE_FOLDER"

DATASET="hpo_metab"
DATASET_NAME="HPO Metab"
RESULT_FOLDER="what_did_we_learn/$DATE_FOLDER/$DATASET"
mkdir -p "$RESULT_FOLDER"
echo "Running Graph Mamba with hops on dataset: $DATASET"
echo "Logging results in: $RESULT_FOLDER" 

mkdir -p "$RESULT_FOLDER/1hop_$DATASET"
python learn_graph_mamba_savegrad.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/1hop_$DATASET" --n-layers 1 --epochs 100 --seqlength 1000 --hops 1
mkdir -p "$RESULT_FOLDER/2hop_$DATASET"
python learn_graph_mamba_savegrad.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/2hop_$DATASET" --n-layers 1 --epochs 100 --seqlength 3000 --hops 2
mkdir -p "$RESULT_FOLDER/3hop_$DATASET"
python learn_graph_mamba_savegrad.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/3hop_$DATASET" --n-layers 1 --epochs 100 --seqlength 5000 --hops 3
python plot_train_val.py --dataset-name "$DATASET_NAME" --result-folder "$RESULT_FOLDER" 
python plot_param_diff_norms.py --dataset-name "$DATASET_NAME" --result-folder "$RESULT_FOLDER" --plot-downstream
python plot_param_grads.py --dataset-name "$DATASET_NAME" --result-folder "$RESULT_FOLDER" --plot-downstream



DATASET="hpo_neruo"
DATASET_NAME="HPO Neuro"
RESULT_FOLDER="what_did_we_learn/$DATE_FOLDER/$DATASET"
mkdir -p "$RESULT_FOLDER"
echo "Running Graph Mamba with hops on dataset: $DATASET"
echo "Logging results in: $RESULT_FOLDER" 

mkdir -p "$RESULT_FOLDER/1hop_$DATASET"
python learn_graph_mamba_savegrad.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/1hop_$DATASET" --n-layers 1 --epochs 100 --seqlength 1000 --hops 1
mkdir -p "$RESULT_FOLDER/2hop_$DATASET"
python learn_graph_mamba_savegrad.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/2hop_$DATASET" --n-layers 1 --epochs 100 --seqlength 3000 --hops 2
mkdir -p "$RESULT_FOLDER/3hop_$DATASET"
python learn_graph_mamba_savegrad.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/3hop_$DATASET" --n-layers 1 --epochs 100 --seqlength 5000 --hops 3
python plot_train_val.py --dataset-name "$DATASET_NAME" --result-folder "$RESULT_FOLDER" 
python plot_param_diff_norms.py --dataset-name "$DATASET_NAME" --result-folder "$RESULT_FOLDER" --plot-downstream
python plot_param_grads.py --dataset-name "$DATASET_NAME" --result-folder "$RESULT_FOLDER" --plot-downstream

DATASET="ppi_bp"
DATASET_NAME="PPI BP"
RESULT_FOLDER="what_did_we_learn/$DATE_FOLDER/$DATASET"
mkdir -p "$RESULT_FOLDER"
echo "Running Graph Mamba with hops on dataset: $DATASET"
echo "Logging results in: $RESULT_FOLDER" 

mkdir -p "$RESULT_FOLDER/1hop_$DATASET"
python learn_graph_mamba_savegrad.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/1hop_$DATASET" --n-layers 1 --epochs 100 --seqlength 1000 --hops 1
mkdir -p "$RESULT_FOLDER/2hop_$DATASET"
python learn_graph_mamba_savegrad.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/2hop_$DATASET" --n-layers 1 --epochs 100 --seqlength 3000 --hops 2
mkdir -p "$RESULT_FOLDER/3hop_$DATASET"
python learn_graph_mamba_savegrad.py --dataset "$DATASET" --logfilename "$RESULT_FOLDER/3hop_$DATASET" --n-layers 1 --epochs 100 --seqlength 5000 --hops 3
python plot_train_val.py --dataset-name "$DATASET_NAME" --result-folder "$RESULT_FOLDER" 
python plot_param_diff_norms.py --dataset-name "$DATASET_NAME" --result-folder "$RESULT_FOLDER" --plot-downstream
python plot_param_grads.py --dataset-name "$DATASET_NAME" --result-folder "$RESULT_FOLDER" --plot-downstream

