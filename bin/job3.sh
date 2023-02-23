#!/bin/bash

#SBATCH --job-name=continuallearning       # job name
#SBATCH --partition=gpu_4                  # queue for the resource allocation.
#SBATCH --time=500:00                     # wall-clock time limit  
#SBATCH --mem=10000                        # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=40                 # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --mail-user=ie2651@partner.kit.edu # notification email address
#SBATCH --gres=gpu:1

module purge                                       # Unload all currently loaded modules.
module load devel/cuda/11.8
source ../ba_env/bin/activate   
configs=("./src/conf/basic_model_stealing/LC_EWC.yaml"
        "./src/conf/basic_model_stealing/LC_IMM.yaml"
        "./src/conf/basic_model_stealing/LC_MAS.yaml"
        "./src/conf/basic_model_stealing/LC_Naive.yaml"
	"./src/conf/basic_model_stealing/LC_Alasso.yaml"
	"./src/conf/basic_model_stealing/BALD_EWC.yaml"
        "./src/conf/basic_model_stealing/BALD_IMM.yaml"
        "./src/conf/basic_model_stealing/BALD_MAS.yaml"
        "./src/conf/basic_model_stealing/BALD_Naive.yaml"
        "./src/conf/basic_model_stealing/BALD_Alasso.yaml"
)
for conf in "${configs[@]}"
do 
    echo "Running $conf with mode CL"
    python ./src/main.py -c $conf -m "CL"
done
deactivate
