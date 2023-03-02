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
module load devel/cuda/11.8
module load mpi/openmpi/4.0
source ../ba_env/bin/activate
files=(
	#"./src/conf/basic_model_stealing/Badge_Naive.yaml"
)
for file in "${files[@]}"
do 
    echo "Running $file with mode AL"
    python ./src/main.py -c $file -m "AL"
done
srun python ./src/parallel_training_demo.py 10 15 --batch_size 128
deactivate



