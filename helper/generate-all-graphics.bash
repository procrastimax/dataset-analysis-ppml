#!/bin/bash

# Dataset Size Experiment
# non-private
#python src/main.py -m cnn -n non-private-dataset-size-comparison -ar 0-7 --xName "# of Samples" --xValues 5000 4000 3000 2000 1000 500 100 50 -ca
#python src/main.py -m cnn -n non-private-dataset-size-comparison -ar 0-7 --xName "# of Samples" --xValues 5000 4000 3000 2000 1000 500 100 50 -ce
## e=30
#python src/main.py -m private_cnn -n private-dataset-size-comparison-eps-30 -ar 0-7 --xName "# of Samples" --xValues 5000 4000 3000 2000 1000 500 100 50 -ce
#python src/main.py -m private_cnn -n private-dataset-size-comparison-eps-30 -ar 0-7 --xName "# of Samples" --xValues 5000 4000 3000 2000 1000 500 100 50 -ca
## e=1
#python src/main.py -m private_cnn -n private-dataset-size-comparison-eps-1 -ar 0-7 --xName "# of Samples" --xValues 5000 4000 3000 2000 1000 500 100 50 -ce
#python src/main.py -m private_cnn -n private-dataset-size-comparison-eps-1 -ar 0-7 --xName "# of Samples" --xValues 5000 4000 3000 2000 1000 500 100 50 -ca

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
