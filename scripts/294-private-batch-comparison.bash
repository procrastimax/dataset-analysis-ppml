#!/bin/bash
#SBATCH -J private-batch-comparison
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --partition=clara
#SBATCH --gres=gpu:v100:1
#SBATCH --time=0-24:00:00
#SBATCH -o logs/%A-%x-%a.out
#SBATCH -e logs/%A-%x-%a.error
#SBATCH --mail-type=FAIL

ds='mnist fmnist cifar10 svhn'
model="private_cnn"
run_name="private-batch-comparison"

# get correct run for array id
case $SLURM_ARRAY_TASK_ID in
    0)
        batch=32
        ;;
    1)
        batch=64
        ;;
    2)
        batch=128
        ;;
    3)
        batch=256
        ;;
    4)
        batch=512
        ;;
esac

echo $run_name "$SLURM_ARRAY_TASK_ID" "-" "$ds" - "$model"

date

srun singularity exec --nv container-dataset-analysis.sif python3.9 src/main.py -d $ds -m $model -r $SLURM_ARRAY_TASK_ID --batch-size $batch -tm -em -n $run_name

date
