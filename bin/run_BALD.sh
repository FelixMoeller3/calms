#!/bin/bash

#SBATCH --job-name=Run_BALD       # job name
#SBATCH --partition=gpu_4_a100                  # queue for the resource allocation.
#SBATCH --time=1200:00                     # wall-clock time limit  
#SBATCH --mem=30000                        # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=2                 # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --mail-user=ie2651@partner.kit.edu # notification email address
#SBATCH --gres=gpu:1

module purge                                       # Unload all currently loaded modules.
module load devel/cuda/11.8
source ../ba_env/bin/activate
files=("./src/conf/basic_model_stealing/BALD_Naive.yaml")
for file in "${files[@]}"
do 
    echo "Running $file with mode MS"
    srun python ./src/main.py -c $file -m "MS"
done
deactivate



