#!/bin/bash
#SBATCH -J non-private-epochs-comparison
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --partition=clara
#SBATCH --gres=gpu:v100:1
#SBATCH --time=0-10:00:00
#SBATCH -o logs/%A-%x-%a.out
#SBATCH -e logs/%A-%x-%a.error
#SBATCH --mail-type=FAIL

ds='mnist fmnist svhn cifar10'
model="cnn"
batch=600
run_name="non-private-epochs-comparison"

# get correct run for array id
case $SLURM_ARRAY_TASK_ID in
        0)
                epochs=5
                ;;
        1)
                epochs=10
                ;;
        2)
                epochs=20
                ;;
        3)
                epochs=30
                ;;
        4)
                epochs=40
                ;;
        5)
                epochs=50
                ;;
        6)
                epochs=60
                ;;
esac

echo "non-private" "$SLURM_ARRAY_TASK_ID" "-" "$ds" - "$model"
srun singularity exec --nv container-dataset-analysis.sif python3.9 src/main.py -d $ds -m $model -r $SLURM_ARRAY_TASK_ID --epochs $epochs --batch-size $batch -n $run_name -tm -em
