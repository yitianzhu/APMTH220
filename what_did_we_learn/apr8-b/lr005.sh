#!/bin/bash
#SBATCH --gres=gpu:volta:1
module load anaconda/2023b
pip show mamba-ssm
python learn.py