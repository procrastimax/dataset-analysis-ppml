#!/bin/bash
#SBATCH -J private-epsilon-comparison
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --partition=clara
#SBATCH --gres=gpu:v100:1
#SBATCH --time=0-20:00:00
#SBATCH -o logs/%A-%x-%a.out
#SBATCH -e logs/%A-%x-%a.error
#SBATCH --mail-type=FAIL

ds='mnist fmnist cifar10 svhn'
epochs=30
model="private_cnn"
batch=512
run_name="private-epsilon-comparison"

# get correct run for array id
case $SLURM_ARRAY_TASK_ID in
        0)
                eps=0.1
                ;;
        1)
                eps=1
                ;;
        2)
                eps=10
                ;;
        3)
                eps=20
                ;;
        4)
                eps=30
                ;;
        5)
                eps=40
                ;;
        6)
                eps=50
                ;;
        7)
                eps=60
                ;;
        8)
                eps=100
                ;;
esac

echo $run_name "$SLURM_ARRAY_TASK_ID" "-" "$ds" - "$model"
srun singularity exec --nv container-dataset-analysis.sif python3.9 src/main.py -e $eps -d $ds -m $model -r $SLURM_ARRAY_TASK_ID --epochs $epochs --batch-size $batch -tm -em -n $run_name
