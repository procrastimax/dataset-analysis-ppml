#!/bin/bash
#SBATCH -J private-learning-single-model-container
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --partition=clara
#SBATCH --gres=gpu:v100:1
#SBATCH --time=0-05:00:00
#SBATCH -o logs/%A-%x-%a.out
#SBATCH -e logs/%A-%x-%a.error
#SBATCH --mail-type=FAIL,END
# #SBATCH --mail-user=

ds='mnist cifar10'

# get correct run for array id
case $SLURM_ARRAY_TASK_ID in
        0)
                l2_norm_clip=1.0
                eps=1.0
                epochs=20
                ;;
        1)
                l2_norm_clip=2.0
                eps=1.0
                epochs=20
                ;;
        2)
                l2_norm_clip=1.0
                eps=1.0
                epochs=30
                ;;
        3)
                l2_norm_clip=1.0
                eps=0.1
                epochs=20
                ;;
        4)
                l2_norm_clip=1.0
                eps=10
                epochs=20
                ;;
esac

batch=200
micro_batch=200

echo "$SLURM_ARRAY_TASK_ID" "-" "$ds" "-" "$eps"
srun singularity exec --nv container-dataset-analysis.sif python3.9 src/main.py -d $ds -r $SLURM_ARRAY_TASK_ID -e $eps -m private_small_cnn -c $l2_norm_clip --epochs $epochs --train-single-model --load-test-single-model --batch-size $batch -b $micro_batch -n "private-learning-changing-params"
