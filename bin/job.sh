#!/bin/bash

#SBATCH --job-name=continuallearning       # job name
#SBATCH --partition=gpu_4                  # queue for the resource allocation.
#SBATCH --time=100:00                     # wall-clock time limit  
#SBATCH --mem=10000                        # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=40                 # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --mail-user=ie2651@partner.kit.edu # notification email address
#SBATCH --gres=gpu:1

module purge                                       # Unload all currently loaded modules.
module load devel/cuda/11.4
source ../ba_env/bin/activate   

python ./src/main.py
deactivate
