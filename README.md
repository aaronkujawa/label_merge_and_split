# Label merge and split

This repository contains the code for the "Label merge-and-split" method.


### Overview

Labels are merged for model training using the following steps:
1. Calculate a label support map based on all label volumes in the training set
1. Calculate a minimum distance matrix between all label pairs according to the label support map
1. Calculate a volume ratio matrix between all label pairs
1. Create an adjacency matrix based on the minimum distance matrix and the volume ratio matrix using a minimum distance
    lower threshold and a volume ratio upper threshold
1. Using a greedy graph coloring algorithm to find label groups that can be merged where every label pair in the group 
    satisfies the minimum distance and volume ratio thresholds
1. Merge the labels in each group into a single merged label

The trained model's merged label predictions are split at inference time using the following steps:
1. A fudged fuzzy prior is calculated for each label based on the label support map using smoothing or a Euclidean
    distance transform
1. Based on the fudged fuzzy prior, one influence region map is calculated for each merged. The influence region map 
    contains regions labeled with the most likely original label of all labels of the corresponding label group.
1. The influence region maps are used to split the merged label predictions into the original labels.

### Example
An example of the full label merge-and-split method is shown in [example_merge_and_split.py](example_scripts%2Fexample_merge_and_split.py)