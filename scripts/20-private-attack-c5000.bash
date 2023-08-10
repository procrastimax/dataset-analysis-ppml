#!/bin/bash
#SBATCH -J private-attack-c5000
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --partition=clara-long
#SBATCH --gres=gpu:v100:1
#SBATCH --time=3-00:00:00
#SBATCH -o logs/%A-%x-%a.out
#SBATCH -e logs/%A-%x-%a.error
#SBATCH --mail-type=FAIL

ds='mnist_c5000 fmnist_c5000 cifar10_c5000 cifar10_c5000_gray svhn_c5000 svhn_c5000_gray'
model="private_cnn"
run_name="private-attack-c5000"

# get correct run for array id
case $SLURM_ARRAY_TASK_ID in
    0)
        eps=1
        ;;
    1)
        eps=30
        ;;
esac

srun singularity exec --nv container-dataset-analysis.sif python3.9 src/main.py -e $eps -d $ds -m $model -r $SLURM_ARRAY_TASK_ID --run-amia-attack --force-model-retrain -n $run_name
