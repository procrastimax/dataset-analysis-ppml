#!/bin/bash
#SBATCH -J private-train-eps-comparison
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --partition=clara
#SBATCH --gres=gpu:v100:1
#SBATCH --time=2-00:00:00
#SBATCH -o logs/%A-%x-%a.out
#SBATCH -e logs/%A-%x-%a.error
#SBATCH --mail-type=FAIL

ds='mnist fmnist cifar10 svhn'
epochs=10
model="private_cnn"
batch=200
run_name="private-train-eps-comparison"

# get correct run for array id
case $SLURM_ARRAY_TASK_ID in
        0)
                eps=1
                ;;
        1)
                eps=10
                ;;
        2)
                eps=20
                ;;
        3)
                eps=30
                ;;
        4)
                eps=40
                ;;
        5)
                eps=50
                ;;
        6)
                eps=60
                ;;
esac

echo $run_name "$SLURM_ARRAY_TASK_ID" "-" "$ds" - "$model"
srun singularity exec --nv container-dataset-analysis.sif python3.9 src/main.py -e $eps -d $ds -m $model -r $SLURM_ARRAY_TASK_ID --epochs $epochs --batch-size $batch --train-model --evaluate-model -n $run_name
