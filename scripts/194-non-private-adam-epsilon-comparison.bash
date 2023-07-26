#!/bin/bash
#SBATCH -J non-private-adam-epsilon-comparison
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --partition=clara
#SBATCH --gres=gpu:v100:1
#SBATCH --time=0-5:00:00
#SBATCH -o logs/%A-%x-%a.out
#SBATCH -e logs/%A-%x-%a.error
#SBATCH --mail-type=FAIL

ds='mnist fmnist svhn cifar10'
model="cnn"
batch=600
run_name="non-private-adam-epsilon-comparison"
epochs=30
lr=0.005

# get correct run for array id
case $SLURM_ARRAY_TASK_ID in
        0)
                ae=1e-9
                ;;
        1)
                ae=1e-7
                ;;
        2)
                ae=1e-6
                ;;
        3)
                ae=1e-5
                ;;
        4)
                ae=1e-4
                ;;
        5)
                ae=1e-3
                ;;
        6)
                ae=1e-2
                ;;
        7)
                ae=1e-1
                ;;
esac

echo "non-private" "$SLURM_ARRAY_TASK_ID" "-" "$ds" - "$model"
srun singularity exec --nv container-dataset-analysis.sif python3.9 src/main.py -d $ds -m $model -r $SLURM_ARRAY_TASK_ID --epochs $epochs --batch-size $batch -n $run_name -tm -em -l $lr -ae $ae
