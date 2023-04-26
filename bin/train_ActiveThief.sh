#!/bin/bash

#SBATCH --job-name=AcTh_Training       # job name
#SBATCH --partition=gpu_4_a100                  # queue for the resource allocation.
#SBATCH --time=30:00                     # wall-clock time limit  
#SBATCH --mem=15000                        # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=2                 # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --mail-user=ie2651@partner.kit.edu # notification email address
#SBATCH --gres=gpu:1

module purge                                       # Unload all currently loaded modules.
module load devel/cuda/11.8
source ../ba_env/bin/activate   
configs=(
	"./src/conf/target_model_training/ATC3_CIFAR-100.yaml"
	#"./src/conf/target_model_training/Resnet_CIFAR-10.yaml"
	#"./src/conf/target_model_training/VGG_Mnist.yaml"
	#"./src/conf/target_model_training/VGG_CIFAR-10.yaml"
        #"./src/conf/basic_model_stealing/Random_Naive.yaml"
	#"./src/conf/basic_model_stealing/BALD_Naive.yaml"
        #"./src/conf/basic_model_stealing/CoreSet_Naive.yaml"
        #"./src/conf/basic_model_stealing/Badge_Naive.yaml"
)
for conf in "${configs[@]}"
do 
    echo "Running $conf with mode TR"
    python ./src/main.py -c $conf -m "TR"
done
deactivate

