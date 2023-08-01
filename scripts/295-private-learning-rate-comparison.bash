#!/bin/bash
#SBATCH -J private-learning-rate-comparison
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
run_name="private-learning-rate-comparison"

# get correct run for array id
case $SLURM_ARRAY_TASK_ID in
        0)
            learning_rate=0.0001
            clip_norm=0.05
                ;;
        1)
            learning_rate=0.001
            clip_norm=0.05
                ;;
        2)
            learning_rate=0.005
            clip_norm=0.05
                ;;
        3)
            learning_rate=0.01
            clip_norm=0.05
                ;;
        4)
            learning_rate=0.05
            clip_norm=0.05
                ;;
        5)
            learning_rate=0.1
            clip_norm=0.05
                ;;


        6)
            learning_rate=0.0001
            clip_norm=0.1
                ;;
        7)
            learning_rate=0.001
            clip_norm=0.1
                ;;
        8)
            learning_rate=0.005
            clip_norm=0.1
                ;;
        9)
            learning_rate=0.01
            clip_norm=0.1
                ;;
        10)
            learning_rate=0.05
            clip_norm=0.1
                ;;
        11)
            learning_rate=0.1
            clip_norm=0.1
                ;;

        12)
            learning_rate=0.0001
            clip_norm=0.5
                ;;
        13)
            learning_rate=0.001
            clip_norm=0.5
                ;;
        14)
            learning_rate=0.005
            clip_norm=0.5
                ;;
        15)
            learning_rate=0.01
            clip_norm=0.5
                ;;
        16)
            learning_rate=0.05
            clip_norm=0.5
                ;;
        17)
            learning_rate=0.1
            clip_norm=0.5
                ;;

        18)
            learning_rate=0.0001
            clip_norm=1.0
                ;;
        19)
            learning_rate=0.001
            clip_norm=1.0
                ;;
        20)
            learning_rate=0.005
            clip_norm=1.0
                ;;
        21)
            learning_rate=0.01
            clip_norm=1.0
                ;;
        22)
            learning_rate=0.05
            clip_norm=1.0
                ;;
        23)
            learning_rate=0.1
            clip_norm=1.0
                ;;
esac

echo $run_name "$SLURM_ARRAY_TASK_ID" "-" "$ds" - "$model"
srun singularity exec --nv container-dataset-analysis.sif python3.9 src/main.py -d $ds -m $model -r $SLURM_ARRAY_TASK_ID -tm -em -n $run_name -c $clip_norm -l $learning_rate
