#!/bin/bash
#SBATCH -J private-clip-search
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --partition=clara
#SBATCH --gres=gpu:v100:1
#SBATCH --time=0-20:00:00
#SBATCH -o logs/%A-%x-%a.out
#SBATCH -e logs/%A-%x-%a.error
#SBATCH --mail-type=FAIL

ds='mnist fmnist cifar10 svhn'
model="private_cnn"
run_name="private-clip-search"
noise=0.0

# get correct run for array id
case $SLURM_ARRAY_TASK_ID in
        0)
            clip_norm=0.0001
                ;;
        1)
            clip_norm=0.001
                ;;
        2)
            clip_norm=0.01
                ;;
        3)
            clip_norm=0.1
                ;;
        4)
            clip_norm=1.0
                ;;
        5)
            clip_norm=10.0
                ;;
        6)
            clip_norm=100.0
                ;;
esac

echo $run_name "$SLURM_ARRAY_TASK_ID" "-" "$ds" - "$model"
srun singularity exec --nv container-dataset-analysis.sif python3.9 src/main.py -d $ds -m $model -r $SLURM_ARRAY_TASK_ID --train-model --evaluate-model -n $run_name -c $clip_norm -np $noise
