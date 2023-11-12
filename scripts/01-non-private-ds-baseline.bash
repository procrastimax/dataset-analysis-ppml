#!/bin/bash
#SBATCH -J non-private-ds-baseline
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --partition=clara
#SBATCH --gres=gpu:v100:1
#SBATCH --time=0-10:00:00
#SBATCH -o logs/%A-%x-%a.out
#SBATCH -e logs/%A-%x-%a.error
#SBATCH --mail-type=FAIL

model="cnn"
run_name="non-private-ds-baseline"

case $SLURM_ARRAY_TASK_ID in
    0)
        ds='mnist fmnist cifar10 svhn'
        ;;
    1)
        ds='cifar10_gray svhn_gray'
        ;;
    2)
        ds='emnist cifar100'
        ;;
esac

srun singularity exec --nv container-dataset-analysis.sif python3.9 src/main.py -d $ds -m $model -r $SLURM_ARRAY_TASK_ID --run-amia-attack --force-model-retrain -n $run_name -tm -em
