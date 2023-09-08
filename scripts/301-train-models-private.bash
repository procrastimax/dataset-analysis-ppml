#!/bin/bash
#SBATCH -J model-train-private
#SBATCH --ntasks=1
#SBATCH --mem=50G
#SBATCH --partition=clara
#SBATCH --gres=gpu:v100:1
#SBATCH --time=1-10:00:00
#SBATCH -o logs/%A-%x-%a.out
#SBATCH -e logs/%A-%x-%a.error
#SBATCH --mail-type=FAIL

model="private_cnn"

case $SLURM_ARRAY_TASK_ID in
    0)
        run_name="private-attack-c5000"
        ;;
    1)
        run_name="private-attack-class-imbalance"
        ;;
    2)
        run_name="private-dataset-size-comparison"
        ;;
    3)
        run_name="private-class-count-comparison"
        ;;
    4)
        run_name="private-class-count-comparison-2"
        ;;
esac

srun singularity exec --nv container-dataset-analysis.sif python3.9 src/main.py -m $model --force-model-retrain -n $run_name -tm -em
