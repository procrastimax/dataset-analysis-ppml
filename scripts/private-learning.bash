#!/bin/bash
#SBATCH -J private-learning-single-model
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --partition=clara
#SBATCH --gres=gpu:v100:1
#SBATCH --time=1-00:00:00
#SBATCH -o logs/%A-%x-%a.out
#SBATCH -e logs/%A-%x-%a.error
#SBATCH --mail-type=FAIL,END
# #SBATCH --mail-user=

# use: sbatch -a 1-4 large-attacker.job
ds='mnist fmnist mnist_c5000 fmnist_c5000 cifar10 cifar10gray'

# get correct run for array id
case $SLURM_ARRAY_TASK_ID in
        0)
                # baseline(?)
                lr=0.001
                l2_norm_clip=1.0
                momentum=0.99
                microbatches=8
                ;;
        1)
                # higher learning rate
                lr=0.1
                l2_norm_clip=1.0
                momentum=0.99
                microbatches=8
                ;;
        2)
                # even more higher learning rate
                lr=0.25
                l2_norm_clip=1.0
                momentum=0.99
                microbatches=8
                ;;
        3)
                # baseline with larger l2 norm clip
                lr=0.001
                l2_norm_clip=0.5
                momentum=0.99
                microbatches=8
                ;;
        4)
                # baselne with smaller l2 norm clip
                lr=0.001
                l2_norm_clip=1.5
                momentum=0.99
                microbatches=8
                ;;
        5)
                # baselien with no momentum
                lr=0.001
                l2_norm_clip=1.0
                momentum=0.0
                microbatches=8
                ;;
        6)
                # baseline with 1 as microbatches
                lr=0.001
                l2_norm_clip=1.0
                momentum=0.99
                microbatches=1
                ;;
        7)
                # baseline with 16 as microbatches
                lr=0.001
                l2_norm_clip=1.0
                momentum=0.99
                microbatches=16
                ;;
esac

echo "$SLURM_ARRAY_TASK_ID" "-" "$ds" "-" "$eps"

module load TensorFlow/2.7.1-foss-2021b-CUDA-11.4.1

source env/bin/activate

srun python src/main.py -d $ds -r $SLURM_ARRAY_TASK_ID --train-single-model -e 0.1 -m private_small_cnn -l $lr -c $l2_norm_clip --momentum $momentum -b $microbatches
