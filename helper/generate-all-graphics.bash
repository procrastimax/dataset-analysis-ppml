#!/bin/bash

# Dataset Size Experiment
# non-private
python src/main.py -m cnn -n non-private-dataset-size-comparison -ar 0-7 --xName "# of Samples" --xValues 5000 4000 3000 2000 1000 500 100 50 -ca
python src/main.py -m cnn -n non-private-dataset-size-comparison -ar 0-7 --xName "# of Samples" --xValues 5000 4000 3000 2000 1000 500 100 50 -ce
# e=30
python src/main.py -m private_cnn -n private-dataset-size-comparison-eps-30 -ar 0-7 --xName "# of Samples" --xValues 5000 4000 3000 2000 1000 500 100 50 -ce
python src/main.py -m private_cnn -n private-dataset-size-comparison-eps-30 -ar 0-7 --xName "# of Samples" --xValues 5000 4000 3000 2000 1000 500 100 50 -ca
# e=1
python src/main.py -m private_cnn -n private-dataset-size-comparison-eps-1 -ar 0-7 --xName "# of Samples" --xValues 5000 4000 3000 2000 1000 500 100 50 -ce
python src/main.py -m private_cnn -n private-dataset-size-comparison-eps-1 -ar 0-7 --xName "# of Samples" --xValues 5000 4000 3000 2000 1000 500 100 50 -ca

# Dataset Class Count Experiment - 47 classes
# non-private
python src/main.py -m cnn -n non-private-class-count-comparison -ar 0-9 --xName "# of Classes" --xValues 47 42 37 32 27 22 17 12 7 3 -ca
python src/main.py -m cnn -n non-private-class-count-comparison -ar 0-9 --xName "# of Classes" --xValues 47 42 37 32 27 22 17 12 7 3 -ce
# e=30
python src/main.py -m private_cnn -n private-class-count-comparison-eps-30 -ar 0-9 --xName "# of Classes" --xValues 47 42 37 32 27 22 17 12 7 3 -ca
python src/main.py -m private_cnn -n private-class-count-comparison-eps-30 -ar 0-9 --xName "# of Classes" --xValues 47 42 37 32 27 22 17 12 7 3 -ce
# e=1
python src/main.py -m private_cnn -n private-class-count-comparison-eps-1 -ar 0-9 --xName "# of Classes" --xValues 47 42 37 32 27 22 17 12 7 3 -ca
python src/main.py -m private_cnn -n private-class-count-comparison-eps-1 -ar 0-9 --xName "# of Classes" --xValues 47 42 37 32 27 22 17 12 7 3 -ce

# Dataset Class Count Experiment - 10 classes
# non-private
python src/main.py -m cnn -n non-private-class-count-comparison-2 -ar 0-7 --xName "# of Classes" --xValues 10 9 8 7 6 5 4 3 -ca
python src/main.py -m cnn -n non-private-class-count-comparison-2 -ar 0-7 --xName "# of Classes" --xValues 10 9 8 7 6 5 4 3 -ce
# e=30
python src/main.py -m private_cnn -n private-class-count-comparison-2-eps-30 -ar 0-7 --xName "# of Classes" --xValues 10 9 8 7 6 5 4 3 -ca
python src/main.py -m private_cnn -n private-class-count-comparison-2-eps-30 -ar 0-7 --xName "# of Classes" --xValues 10 9 8 7 6 5 4 3 -ce
# e=1
python src/main.py -m private_cnn -n private-class-count-comparison-2-eps-1 -ar 0-7 --xName "# of Classes" --xValues 10 9 8 7 6 5 4 3 -ca
python src/main.py -m private_cnn -n private-class-count-comparison-2-eps-1 -ar 0-7 --xName "# of Classes" --xValues 10 9 8 7 6 5 4 3 -ce

# Dataset Imbalance Experiment - Normal Mode
# non-private
python src/main.py -m cnn -n non-private-imbalance-comparison-n -ar 0-4 --xName "Imbalance Factor (Normal Mode)" --xValues 0.0 0.1 0.3 0.6 0.9 -ca
python src/main.py -m cnn -n non-private-imbalance-comparison-n -ar 0-4 --xName "Imbalance Factor (Normal Mode)" --xValues 0.0 0.1 0.3 0.6 0.9 -ce
# e=30
python src/main.py -m private_cnn -n private-imbalance-comparison-n-eps-30 -ar 0-4 --xName "Imbalance Factor (Normal Mode)" --xValues 0.0 0.1 0.3 0.6 0.9 -ca
python src/main.py -m private_cnn -n private-imbalance-comparison-n-eps-30 -ar 0-4 --xName "Imbalance Factor (Normal Mode)" --xValues 0.0 0.1 0.3 0.6 0.9 -ce
# e=1
python src/main.py -m private_cnn -n private-imbalance-comparison-n-eps-1 -ar 0-4 --xName "Imbalance Factor (Normal Mode)" --xValues 0.0 0.1 0.3 0.6 0.9 -ca
python src/main.py -m private_cnn -n private-imbalance-comparison-n-eps-1 -ar 0-4 --xName "Imbalance Factor (Normal Mode)" --xValues 0.0 0.1 0.3 0.6 0.9 -ce

# Dataset Imbalance Experiment - Linear Mode
# non-private
python src/main.py -m cnn -n non-private-imbalance-comparison-l -ar 0-4 --xName "Imbalance Factor (Linear Mode)" --xValues 0.0 0.1 0.3 0.6 0.9 -ca
python src/main.py -m cnn -n non-private-imbalance-comparison-l -ar 0-4 --xName "Imbalance Factor (Linear Mode)" --xValues 0.0 0.1 0.3 0.6 0.9 -ce
# e=30
python src/main.py -m private_cnn -n private-imbalance-comparison-l-eps-30 -ar 0-4 --xName "Imbalance Factor (Linear Mode)" --xValues 0.0 0.1 0.3 0.6 0.9 -ca
python src/main.py -m private_cnn -n private-imbalance-comparison-l-eps-30 -ar 0-4 --xName "Imbalance Factor (Linear Mode)" --xValues 0.0 0.1 0.3 0.6 0.9 -ce
# e=1
python src/main.py -m private_cnn -n private-imbalance-comparison-l-eps-1 -ar 0-4 --xName "Imbalance Factor (Linear Mode)" --xValues 0.0 0.1 0.3 0.6 0.9 -ca
python src/main.py -m private_cnn -n private-imbalance-comparison-l-eps-1 -ar 0-4 --xName "Imbalance Factor (Linear Mode)" --xValues 0.0 0.1 0.3 0.6 0.9 -ce

# Data Level Investigation
python src/main.py -m cnn -n evaluation-basic-ds -ar 0-2 --xName "Privacy Budget" --xValues inf 30 1 -ca
python src/main.py -m cnn -n evaluation-basic-ds -ar 0-2 --xName "Privacy Budget" --xValues inf 30 1 -ce
python src/main.py -m cnn -n evaluation-color-ds -ar 0-2 --xName "Privacy Budget" --xValues inf 30 1 -ca
python src/main.py -m cnn -n evaluation-color-ds -ar 0-2 --xName "Privacy Budget" --xValues inf 30 1 -ce
python src/main.py -m cnn -n evaluation-complete-ds -ar 0-2 --xName "Privacy Budget" --xValues inf 30 1 -ca
python src/main.py -m cnn -n evaluation-complete-ds -ar 0-2 --xName "Privacy Budget" --xValues inf 30 1 -ce
