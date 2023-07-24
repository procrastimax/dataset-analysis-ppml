#!/bin/bash
#SBATCH -J private-train-epochs-comparison
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
batch=200
run_name="private-train-epochs-comparison"

# get correct run for array id
case $SLURM_ARRAY_TASK_ID in
        0)
                eps=1
                epochs=5
                ;;
        1)
                eps=1
                epochs=10
                ;;
        2)
                eps=1
                epochs=20
                ;;
        3)
                eps=1
                epochs=30
                ;;
        4)
                eps=30
                epochs=5
                ;;
        5)
                eps=30
                epochs=10
                ;;
        6)
                eps=30
                epochs=20
                ;;
        7)
                eps=30
                epochs=30
                ;;
esac

echo $run_name "$SLURM_ARRAY_TASK_ID" "-" "$ds" - "$model"
srun singularity exec --nv container-dataset-analysis.sif python3.9 src/main.py -e $eps -d $ds -m $model -r $SLURM_ARRAY_TASK_ID --epochs $epochs --batch-size $batch --train-model --evaluate-model -n $run_name
