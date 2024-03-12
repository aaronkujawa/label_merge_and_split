import os
from multiprocessing import Pool

import networkx as nx
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from skimage import segmentation
from tqdm import tqdm


def get_label_to_channel_mapping(label_paths):
    """
    This function creates a mapping from not necessarily consecutive input labels to consecutive channel numbers.
    This is useful for creating a one-hot encoding of the labels.
    The mapping is created by finding all unique labels in the label_paths and then sorting them.
    :param label_paths:
    :return: label_to_channel_map: a dictionary that maps each label to a channel number
    """

    # find all unique labels in the label_paths
    all_unique_labels = set()
    label_to_channel_map = {}
    for label_path in label_paths:
        unique_labels = np.unique(nib.load(label_path).get_fdata())
        all_unique_labels.update(unique_labels)

    # sort the unique labels
    all_unique_labels = sorted(list(all_unique_labels))

    # create a mapping from label to channel
    for i, label in enumerate(all_unique_labels):
        label_to_channel_map[label] = i

    return label_to_channel_map


def get_label_support(label_paths, label_to_channel_map, save_path=None):  # equation 1
    """
    This function calculates the label support for each label in found in the label_paths data arrays, i.e. it counts
    the number of times each label appears in the label_paths data arrays at each voxel.
    This requires that labels were co-registered to the same space.
    :param label_paths: list of paths to the label files
    :param label_to_channel_map: a dictionary that maps each label to a consecutive channel number
    :param save_path: path to save the label support
    :return: label_support: an array of shape (num_labels, *data_shape) that contains the label support for each label
    """
    for i, parc_path in enumerate(tqdm(label_paths)):
        parc_data = torch.tensor(nib.load(parc_path).get_fdata().astype(int), device="cuda")

        if i == 0:
            nb_labels = len(label_to_channel_map)
            label_support = torch.zeros((nb_labels,) + parc_data.shape, device="cuda").float()

        for lab in torch.unique(parc_data):
            to_add_to_channel = parc_data == lab

            lab = lab.to('cpu').item()
            channel = label_to_channel_map[lab]
            label_support[channel] += to_add_to_channel

    if save_path:
        # convert to int16 to save space (max value allowed: 32767)
        assert (torch.max(label_support) < 32767)
        label_support = label_support.to(torch.int16)
        print(f"Saving label support to {save_path}")
        torch.save(label_support, save_path)

    return label_support


def get_distance(args):
    """
    Helper function to calculate the minimum distance between two labels in parallel
    :param args: a list containing the first label, the second label and the label support array
    :return: a list containing the first label, the second label and the minimum distance between the two labels in the
    label support
    """
    # print(label_pair, flush=True)
    label1 = args[0]
    label2 = args[1]
    label_support = args[2]

    spacing = [1, 1, 1]

    # get coordinates of boundary voxels for both labels
    binary_img_label1 = label_support[label1].cpu().numpy()
    binary_img_label2 = label_support[label2].cpu().numpy()

    if np.count_nonzero(binary_img_label1) and np.count_nonzero(binary_img_label2):  # check if both labels exist
        boundary_coordinates_label1 = np.array(list(zip(*np.where(binary_img_label1))))
        boundary_coordinates_label2 = np.array(list(zip(*np.where(binary_img_label2))))

        # calculate the minimum distances for each point in label1 to any point in label2
        min_dists, _ = cKDTree(boundary_coordinates_label1 * np.array(spacing)).query(
            boundary_coordinates_label2 * np.array(spacing), 1)

        # further, get the mimimum distance from any point in label1 to any point in label2
        dist = np.min(min_dists)
    else:  # otherwise return nan
        dist = np.nan

    # print([label1, label2, dist], flush=True)
    return [label1, label2, dist]


def get_distance_matrix(label_support):  # equation 2
    """
    This function calculates the minimum distances between original labels as observed in the label support.
    This information is encoded in the distance matrix  whose entries are the minimum Euclidean distance between any
    two voxels of original label l1 and l2 taken from any two training volumes in MNI space
    :param label_support: an array of shape (num_labels, *data_shape) that contains the label support for each label
    :return: distance_matrix: an array of shape (num_labels, num_labels) that contains the minimum distances between
    original labels
    """

    # in the label support, keep only the boundaries of the labels to speed up the distance calculation
    for c in range(label_support.shape[0]):
        mask = label_support[c] > 0
        mask = mask.cpu().numpy()
        if np.count_nonzero(mask):
            mask_boundary = segmentation.find_boundaries(mask, connectivity=1, mode='inner', background=0)
        else:
            mask_boundary = mask
        label_support[c] = torch.tensor(mask_boundary, device="cuda")

    print(f"{label_support.shape=}")

    # label support needs to be on the cpu
    if torch.is_tensor(label_support):
        label_support = label_support.cpu()

    # get the inputs for the helper function (first label, second label, label support)
    input_args = []
    nb_labels = len(label_support)
    for label1 in range(0, nb_labels):
        for label2 in range(label1 + 1, nb_labels):
            input_args.append([label1, label2, label_support])

    # calculate distances in parallel
    with Pool(10) as p:
        min_dists = list(tqdm(p.imap(get_distance, input_args), total=len(input_args)))

    # create distance matrix
    min_dist_mat = np.zeros([nb_labels, nb_labels])
    for dist in tqdm(min_dists):
        min_dist_mat[dist[0], dist[1]] = dist[2]
        min_dist_mat[dist[1], dist[0]] = dist[2]  # make symmetric

    # set diagonal to infinity
    nb_labels = min_dist_mat.shape[0]
    min_dist_mat = min_dist_mat + np.diag([np.Inf] * nb_labels)

    return min_dist_mat


def get_average_volume_ratio_matrix(label_support):
    """
    The average volume ratio matrix between two labels l1 and l2 is calculated by dividing the larger average
    label volume by the smaller average label volume.
    :param label_support: an array of shape (num_labels, num_labels) that contains the average volume
    :return: average_vol_ratio_matrix: an array of shape (num_labels, num_labels) that contains the average volume ratio
    """

    # loop through all label paths and calculate the average volume matrix for each label pair
    nb_labels = len(label_support)
    average_vol_ratio_matrix = np.zeros([nb_labels, nb_labels])
    for i in range(0, nb_labels):
        for j in range(i + 1, nb_labels):
            sum_label1 = torch.sum(label_support[i])
            sum_label2 = torch.sum(label_support[j])
            average_vol_ratio_matrix[i, j] = max(sum_label1, sum_label2) / min(sum_label1, sum_label2)
            average_vol_ratio_matrix[j, i] = average_vol_ratio_matrix[i, j]  # make symmetric

    return average_vol_ratio_matrix


def get_adjacency_matrix(distance_matrix, distance_threshold, average_vol_ratio_matrix, vol_ratio_threshold):
    """
    The adjacency matrix is calculated by combining the distance matrix and the average volume ratio matrix.
    The adjacency matrix represents which nodes/labels are connected in the graph. Connected nodes/labels cannot be
     merged by the graph coloring algorithm.
    :param distance_matrix: an array of shape (num_labels, num_labels) that contains the minimum distances between
    original labels
    :param distance_threshold: a lower threshold for the distance matrix
    :param average_vol_ratio_matrix: an array of shape (num_labels, num_labels) that contains the average volume ratio
    :param vol_ratio_threshold: an upper threshold for the average volume ratio matrix
    :return: adjacency_matrix: an array of shape (num_labels, num_labels) that contains the adjacency matrix
    """

    # create adjacency matrix
    adjacency_matrix = np.ones_like(distance_matrix)
    adjacency_matrix[(distance_matrix > distance_threshold) & (average_vol_ratio_matrix < vol_ratio_threshold)] = 0

    return adjacency_matrix


def get_orig_to_merged_label_map(adjacency_matrix, label_to_channel_map, dont_merge_labels=[0,]):
    """
    This function creates a mapping from the original labels to the merged labels.
    The mapping is calculated using the graph coloring algorithm with the adjacency matrix. The graph coloring algorithm
    assigns a color to each node/label such that no two connected nodes/labels have the same color. Connected nodes/labels
    in the adjacency matrix cannot be merged.
    :param adjacency_matrix: an array of shape (num_labels, num_labels) that contains the adjacency matrix
    :param label_to_channel_map: a dictionary that maps each original label to the consecutive channel number used in the
    adjacency matrix
    :param dont_merge_labels: a list of labels that should not be merged, for example the background label, because
    the preprocessing might depend on the background image intensities
    :return: orig_to_merged_label_map: a dictionary that maps each original label to a merged label
    """

    # modify the adjacency matrix such that the dont_merge_labels are not merged
    for label in dont_merge_labels:
        channel = label_to_channel_map[label]
        adjacency_matrix[channel, :] = 1  # connected nodes will not be merged
        adjacency_matrix[:, channel] = 1
        adjacency_matrix[channel, channel] = 0  # smallest_last strategy does not work with 1s on the diagonal

    # convert adjacency matrix to graph
    graph = nx.from_numpy_array(adjacency_matrix)

    # apply the greedy color graph colouring algorithm
    strategies = \
        ['largest_first',
         'random_sequential',
         'smallest_last',
         'independent_set',
         'connected_sequential_bfs',
         'connected_sequential_dfs',
         'saturation_largest_first']

    # try the different strategies and choose the strategy that gives the smallest number of combined labels
    best_strategy = None
    for s in strategies:
        channel_to_merged_label_dict = nx.greedy_color(graph, strategy=s, interchange=False)

        channel_to_merged_label_dict = dict(sorted(channel_to_merged_label_dict.items()))
        nb_comb_labels = len(np.unique(list(channel_to_merged_label_dict.values())))
        print(s, nb_comb_labels)
        if best_strategy is None or nb_comb_labels < best_nb_comb_labels:
            best_strategy = s
            best_nb_comb_labels = nb_comb_labels
            best_old_to_new_label_dict = channel_to_merged_label_dict

    print("Best strategy: ", best_strategy)
    print("Number of combined labels: ", best_nb_comb_labels)

    channel_to_merged_label_dict = best_old_to_new_label_dict

    # create a mapping from the original labels to the merged labels
    orig_to_merged_label_map = {}
    for orig_label, channel in label_to_channel_map.items():
        orig_to_merged_label_map[orig_label] = channel_to_merged_label_dict[channel]

    # if the original background label is not mapped to 0, then we need to remap the merged labels
    current_merged_background_label = orig_to_merged_label_map[0]

    if current_merged_background_label != 0:
        # map the labels that are currently mapped to 0 to the current_merged_background_label and vice versa
        for orig_label, merged_label in orig_to_merged_label_map.items():
            if merged_label == 0:
                orig_to_merged_label_map[orig_label] = current_merged_background_label
            elif merged_label == current_merged_background_label:
                orig_to_merged_label_map[orig_label] = 0

        orig_to_merged_label_map[0] = 0

    return orig_to_merged_label_map


def get_merged_label_dataframe(label_paths,
                               label_to_name_csv_path,
                               distance_threshold=1.0,
                               volume_ratio_threshold=3.5,
                               dont_merge_labels=[0,],
                               output_dir=None,
                               debug=False):
    """
    This function creates a pandas dataframe that contains the original labels, channels, merged labels and label names,
    merged label names and a list of original labels that are merged into the merged label.
    :param label_paths: list of paths to the label files
    :param label_to_name_csv_path: path to the csv file that contains the label names
    :param distance_threshold: a lower threshold for the distance matrix
    :param volume_ratio_threshold: an upper threshold for the average volume ratio matrix
    :param dont_merge_labels: a list of labels that should not be merged, for example the background label, because
    the preprocessing might depend on the background image intensities
    :param debug: if True, only a subset of the labels is used for testing
    :return: label_dataframe: a pandas dataframe that contains the original labels, channels, merged labels and label names,
    merged label names and a list of original labels that are merged into the merged label
    """

    if output_dir:
        label_support_save_path = os.path.join(output_dir, "label_support.pt")

    label_to_name_map = pd.read_csv(label_to_name_csv_path, index_col=0).to_dict()["name"]
    label_to_channel_map = get_label_to_channel_mapping(label_paths)
    label_support = get_label_support(label_paths, label_to_channel_map, save_path=label_support_save_path)

    if debug:
        # reduce number of labels
        nb_test_labels = 10
        label_support = label_support[:nb_test_labels, ...]
        label_to_channel_map = {label: channel for label, channel in label_to_channel_map.items() if channel < nb_test_labels}
        label_to_name_map = {label: name for label, name in label_to_name_map.items() if label in label_to_channel_map}

    distance_matrix = get_distance_matrix(label_support)
    average_volume_ratio_matrix = get_average_volume_ratio_matrix(label_support)
    adjacency_matrix = get_adjacency_matrix(distance_matrix, distance_threshold, average_volume_ratio_matrix, volume_ratio_threshold)
    orig_to_merged_label_map = get_orig_to_merged_label_map(adjacency_matrix, label_to_channel_map, dont_merge_labels)

    # create a pandas dataframe that contains the original labels, channels, merged labels and label names
    label_dataframe = pd.DataFrame(columns=["label", "channel", "merged_label", "name", "merged_label_name", "labels_in_merge"])

    assert len(label_to_channel_map) == len(orig_to_merged_label_map) == len(label_to_name_map), \
        (f"Lengths of label_to_channel_map, orig_to_merged_label_map and label_to_name_map do not match, "
         f"{len(label_to_channel_map)=}, {len(orig_to_merged_label_map)=}, {len(label_to_name_map)=}")

    label_dataframe["label"] = list(label_to_channel_map.keys())
    label_dataframe["channel"] = list(label_to_channel_map.values())
    label_dataframe["merged_label"] = [orig_to_merged_label_map[label] for label in label_to_channel_map.keys()]
    label_dataframe["name"] = [label_to_name_map[label] for label in label_to_channel_map.keys()]
    label_dataframe["merged_label_name"] = \
        ["merged " + str(len(label_dataframe[label_dataframe["merged_label"] == merged_label])) + " labels: " +
         "+".join(label_dataframe[label_dataframe["merged_label"] == merged_label]["name"].tolist()) for merged_label
         in label_dataframe["merged_label"]]
    label_dataframe["labels_in_merge"] = \
        [label_dataframe[label_dataframe["merged_label"] == merged_label]["label"].tolist() for merged_label
         in label_dataframe["merged_label"]]

    if output_dir is not None:
        label_dataframe.to_csv(os.path.join(output_dir, "merged_labels.csv"), index=False)

    return label_dataframe


def merge_label_volumes(label_paths_in, label_paths_out, merged_labels_csv_path):
    """
    This function merges the label volumes according to the merged_labels_csv_path and saves the merged label volumes
    to label_paths_out.
    :param label_paths_in: list of paths to the label files
    :param label_paths_out: list of paths to the output label files
    :param merged_labels_csv_path: path to the csv file that contains the merged labels
    """
    orig_to_merged_label_map = pd.read_csv(merged_labels_csv_path, index_col='label').to_dict()["merged_label"]

    for label_path_in, label_path_out in zip(label_paths_in, label_paths_out):
        # load data
        label_nii = nib.load(label_path_in)
        label_data = label_nii.get_fdata()

        # merge labels using torch if cuda is available
        if torch.cuda.is_available():
            label_data_merged = torch.tensor(label_data, device="cuda")
            for orig_label, merged_label in orig_to_merged_label_map.items():
                label_data_merged[label_data == orig_label] = merged_label
            label_data_merged = label_data_merged.cpu().numpy()
        else:
            # merge labels using vectorize
            label_data_merged = np.vectorize(orig_to_merged_label_map.get, otypes=[np.float32])(label_data)

        # save file
        label_data_merged_nii = nib.Nifti1Image(label_data_merged, label_nii.affine)
        nib.save(label_data_merged_nii, label_path_out)

        print(f"Saved merged label volume to {label_path_out}")