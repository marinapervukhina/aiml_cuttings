#!/bin/bash
#SBATCH --account=<PROJECT_ID>
#SBATCH --time=52:00:00
#SBATCH --mem=128GB
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

cd $SLURM_SUBMIT_DIR

module load tensorflow
python model_training_flow.py --batch 128 --epoch 200
