#!/bin/bash

#SBATCH --job-name=Parallel_Test       # job name
#SBATCH --partition=gpu_4_a100                  # queue for the resource allocation.
#SBATCH --time=10:00                     # wall-clock time limit  
#SBATCH --mem=100000                        # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --mail-user=ie2651@partner.kit.edu # notification email address
#SBATCH --gres=gpu:3
#SBATCH --ntasks=1
#SBARCH --ntasks-per-gpu=1
#SBATCH --cpus-per-task=3

module purge                                       # Unload all currently loaded modules.
module load devel/cuda/11.8
source ../ba_env/bin/activate
files=("./src/conf/basic_model_stealing/Badge_Naive.yaml"
	"./src/conf/basic_model_stealing/Badge_Naive.yaml"
	"./src/conf/basic_model_stealing/Badge_Naive.yaml"
)
for file in "${files[@]}"
do 
    echo "Running $file with mode AL"
    #srun -c1 python ./src/main.py -c $file -m "AL" &
    srun -c 1 --exclusive python -c "print('hello world!')" &
done
wait
deactivate


