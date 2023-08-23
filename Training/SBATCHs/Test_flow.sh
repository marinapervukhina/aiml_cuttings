#!/bin/bash
#SBATCH --account=<PROJECT_ID>
#SBATCH --time=02:15:00
#SBATCH --mem=128GB
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2

cd $SLURM_SUBMIT_DIR

module load tensorflow

 
python testing_model.py --rocks ALL --directory . --model_to_load train_model.epoch30-loss0.17.hdf5
