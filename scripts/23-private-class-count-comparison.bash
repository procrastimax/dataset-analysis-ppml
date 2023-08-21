#!/bin/bash
#SBATCH -J private-class-count-comparison
#SBATCH --ntasks=1
#SBATCH --mem=50G
#SBATCH --partition=clara
#SBATCH --gres=gpu:v100:1
#SBATCH --time=2-00:00:00
#SBATCH -o logs/%A-%x-%a.out
#SBATCH -e logs/%A-%x-%a.error
#SBATCH --mail-type=FAIL

run_name="private-class-count-comparison"
model="private_cnn"

case $SLURM_ARRAY_TASK_ID in
    0)
        ds='emnist_n47 cifar100_n47'
        eps=1
        ;;
    1)
        ds='emnist_n42 cifar100_n42'
        eps=1
        ;;
    2)
        ds='emnist_n37 cifar100_n37'
        eps=1
        ;;
    3)
        ds='emnist_n32 cifar100_n32'
        eps=1
        ;;
    4)
        ds='emnist_n27 cifar100_n27'
        eps=1
        ;;
    5)
        ds='emnist_n22 cifar100_n22'
        eps=1
        ;;
    6)
        ds='emnist_n17 cifar100_n17'
        eps=1
        ;;
    7)
        ds='emnist_n12 cifar100_n12'
        eps=1
        ;;
    8)
        ds='emnist_n7 cifar100_n7'
        eps=1
        ;;
    9)
        ds='emnist_n3 cifar100_n3'
        eps=1
        ;;
    10)
        ds='emnist_n47 cifar100_n47'
        eps=30
        ;;
    11)
        ds='emnist_n42 cifar100_n42'
        eps=30
        ;;
    12)
        ds='emnist_n37 cifar100_n37'
        eps=30
        ;;
    13)
        ds='emnist_n32 cifar100_n32'
        eps=30
        ;;
    14)
        ds='emnist_n27 cifar100_n27'
        eps=30
        ;;
    15)
        ds='emnist_n22 cifar100_n22'
        eps=30
        ;;
    16)
        ds='emnist_n17 cifar100_n17'
        eps=30
        ;;
    17)
        ds='emnist_n12 cifar100_n12'
        eps=30
        ;;
    18)
        ds='emnist_n7 cifar100_n7'
        eps=30
        ;;
    19)
        ds='emnist_n3 cifar100_n3'
        eps=30
        ;;
esac

srun singularity exec --nv container-dataset-analysis.sif python3.9 src/main.py -d $ds -m $model -r $SLURM_ARRAY_TASK_ID --run-amia-attack --force-model-retrain -n $run_name -e $eps
