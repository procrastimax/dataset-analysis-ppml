#!/bin/bash
#SBATCH -J non-private-class-count-comparison
#SBATCH --ntasks=1
#SBATCH --mem=50G
#SBATCH --partition=clara
#SBATCH --gres=gpu:v100:1
#SBATCH --time=1-00:00:00
#SBATCH -o logs/%A-%x-%a.out
#SBATCH -e logs/%A-%x-%a.error
#SBATCH --mail-type=FAIL

run_name="non-private-class-count-comparison"
model="cnn"

case $SLURM_ARRAY_TASK_ID in
    0)
        ds='emnist_n47 cifar100_n47'
        ;;
    1)
        ds='emnist_n42 cifar100_n42'
        ;;
    2)
        ds='emnist_n37 cifar100_n37'
        ;;
    3)
        ds='emnist_n32 cifar100_n32'
        ;;
    4)
        ds='emnist_n27 cifar100_n27'
        ;;
    5)
        ds='emnist_n22 cifar100_n22'
        ;;
    6)
        ds='emnist_n17 cifar100_n17'
        ;;
    7)
        ds='emnist_n12 cifar100_n12'
        ;;
    8)
        ds='emnist_n7 cifar100_n7'
        ;;
    9)
        ds='emnist_n3 cifar100_n3'
        ;;
esac

srun singularity exec --nv container-dataset-analysis.sif python3.9 src/main.py -d $ds -m $model -r $SLURM_ARRAY_TASK_ID --run-amia-attack --force-model-retrain -n $run_name
