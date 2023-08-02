#!/bin/bash
#SBATCH -J private-microbatch-comparison
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
batch=512
run_name="private-microbatch-comparison"

# get correct run for array id
case $SLURM_ARRAY_TASK_ID in
        0)
            microbatch=1
                ;;
        1)
            microbatch=8
                ;;
        2)
            microbatch=16
                ;;
        3)
            microbatch=32
                ;;
        4)
            microbatch=64
                ;;
        5)
            microbatch=128
                ;;
        6)
            microbatch=256
                ;;
        7)
            microbatch=512
                ;;
esac

echo $run_name "$SLURM_ARRAY_TASK_ID" "-" "$ds" - "$model"
srun singularity exec --nv container-dataset-analysis.sif python3.9 src/main.py -d $ds -m $model -r $SLURM_ARRAY_TASK_ID --batch-size $batch -tm -em -n $run_name -b $microbatch
