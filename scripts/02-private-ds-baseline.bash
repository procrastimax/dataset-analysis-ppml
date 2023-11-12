#!/bin/bash
#SBATCH -J private-ds-baseline
#SBATCH --ntasks=1
#SBATCH --mem=50G
#SBATCH --partition=clara
#SBATCH --gres=gpu:v100:1
#SBATCH --time=1-12:00:00
#SBATCH -o logs/%A-%x-%a.out
#SBATCH -e logs/%A-%x-%a.error
#SBATCH --mail-type=FAIL

model="private_cnn"
run_name="private-ds-baseline"

case $SLURM_ARRAY_TASK_ID in
    0)
        ds='mnist fmnist cifar10 svhn'
        eps=1
        ;;
    1)
        ds='cifar10_gray svhn_gray'
        eps=1
        ;;
    2)
        ds='emnist cifar100'
        eps=1
        ;;
    3)
        ds='mnist fmnist cifar10 svhn'
        eps=30
        ;;
    4)
        ds='cifar10_gray svhn_gray'
        eps=30
        ;;
    5)
        ds='emnist cifar100'
        eps=30
        ;;
esac

srun singularity exec --nv container-dataset-analysis.sif python3.9 src/main.py -d $ds -m $model -r $SLURM_ARRAY_TASK_ID --run-amia-attack --force-model-retrain -n $run_name -tm -em --epsilon $eps
