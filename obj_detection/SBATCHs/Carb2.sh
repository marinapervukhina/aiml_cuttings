#!/bin/bash
#SBATCH --account <INSERT PROJECT ID>
#SBATCH --time=06:00:00
#SBATCH --mem=64GB
#SBATCH --ntasks-per-node=16


cd $SLURM_SUBMIT_DIR

module load tensorflow 

python obj_dect_2.py --input_path <PATH_TO_ARW_IMAGE>  --rf 1.0 --rocks <SELECT_ROCK_TYPE>  --np 16 

