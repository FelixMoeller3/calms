#!/bin/bash

#SBATCH --job-name=Parallel_Test       # job name
#SBATCH --partition=gpu_4_a100                  # queue for the resource allocation.
#SBATCH --time=10:00                     # wall-clock time limit  
#SBATCH --mem=100000                        # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --mail-user=ie2651@partner.kit.edu # notification email address
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBARCH --ntasks-per-gpu=1

module purge                                       # Unload all currently loaded modules.
module load compiler/gnu/10.2
module load devel/cuda/10.2

source ../ba_env/bin/activate
unset SLURM_NTASKS_PER_TRES

srun python ./src/parallel_training_demo2.py 5 --batch_size 64

deactivate



