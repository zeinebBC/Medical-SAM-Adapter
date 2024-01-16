#!/bin/bash
#
# CPU resources
#
# GPU resources
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=16gb
#
# display / output
#SBATCH --output=train_sparse_local-%j.out
#SBATCH --error=train_sparse_local-%j.err
#
#SBATCH --job-name=train_sparse_local

source /home/zozchaab/anaconda3/etc/profile.d/conda.sh
conda activate msa_env

BASE_DIR=/home/zozchaab/Medical-SAM-Adapter
pushd $BASE_DIR

git add .
git commit -m "fix mask loading"
git push 
popd
