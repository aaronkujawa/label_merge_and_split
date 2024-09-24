import os
from glob import glob

import torch
from natsort import natsorted
import nibabel as nib
import pandas as pd
import numpy as np
from monai.transforms import AffineGrid
from math import pi
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/aaron/Dropbox/KCL/Tools/CustomScripts')
from calculate_dice import get_dice

from labelmergeandsplit.labelreg_utils import get_coms, get_covs, affine_transform_image, optimize_affine_labelreg, \
    affine_transform_influence_regions
from labelmergeandsplit.merging_utils import get_merged_label_dataframe, merge_label_volumes, map_labels_in_volume
from labelmergeandsplit.splitting_utils import get_fuzzy_prior_fudged, split_merged_labels_paths, split_merged_label, \
    get_influence_regions
from utils.plot_matrix_slices import plot_matrix_slices

from config.config import PROJ_ROOT

''' 
This script demonstrates how to split labels that are not registered to the same space as the influence regions.

'''

# set the random seed
np.random.seed(0)


def get_random_affine(img_shape):
    shape = img_shape

    # example affine matrix
    # shift to image center
    R_origin_to_center = np.array([
        [1, 0, 0, -shape[0] / 2],
        [0, 1, 0, -shape[1] / 2],
        [0, 0, 1, -shape[2] / 2],
        [0, 0, 0, 1],
    ])

    affineGrid = AffineGrid(
        rotate_params=[pi / 3, pi / 5, pi / 2],  # Sequence[float] | float | None = None,
        shear_params=[0.1] * 6,  # Sequence[float] | float | None = None,
        translate_params=[-0, -0, 0],  # Sequence[float] | float | None = None,
        scale_params=None,  # Sequence[float] | float | None = None,
        device='cpu',  # np.device | None = None,
        dtype=np.float32,  # DtypeLike = np.float32,
        align_corners=False,  # bool = False,
        affine=None,  # NdarrayOrTensor | None = None,
        lazy=False,  # bool = False,
    )
    _, R_rot = affineGrid(spatial_size=(64, 64, 64))
    R_rot = R_rot.cpu().numpy()

    # shift back to original position
    R_origin_to_corner = np.array([
        [1, 0, 0, shape[0] / 2],
        [0, 1, 0, shape[1] / 2],
        [0, 0, 1, shape[2] / 2],
        [0, 0, 0, 1],
    ])

    R = R_origin_to_corner @ R_rot @ R_origin_to_center

    return R


def get_network_output_from_registered_image(label_path_unmerged, pad):
    # load an original label image
    label_nii_unmerged = nib.load(label_path_unmerged)
    label_data_unmerged = label_nii_unmerged.get_fdata()

    # pad the label image to avoid cropping of foreground after transformation
    label_data_unmerged = np.pad(label_data_unmerged, pad, mode='constant', )

    # merge the labels
    label_data_merged = map_labels_in_volume(label_data_unmerged, label_to_merged_label_mapping)

    # get an example affine that represents the ground truth transformation (to be estimated)
    R_grtr = get_random_affine(label_data_unmerged.shape)
    # R_grtr = np.diag([1, 1, 1, 1])

    # transform the label image
    transformed_label_unmerged = torch.tensor(
        affine_transform_image(torch.tensor(label_data_unmerged), torch.tensor(np.linalg.inv(R_grtr))))

    # put on GPU
    transformed_label_unmerged = transformed_label_unmerged.to("cuda")

    # merge the labels
    transformed_label_merged = map_labels_in_volume(transformed_label_unmerged, label_to_merged_label_mapping)

    return transformed_label_merged, transformed_label_unmerged, label_data_merged, R_grtr


def plot_results(image_title_pairs):
    # plot the images
    fig, axes = plt.subplots(1, len(image_title_pairs), figsize=(3 * len(image_title_pairs), 6))

    for i, (image, title) in enumerate(image_title_pairs):
        image = image.cpu().numpy() if isinstance(image, torch.Tensor) else image
        axes[i].imshow(image[:, :, image.shape[2] // 2])
        axes[i].set_title(title)
    plt.show()


# # label support is needed to get the influence regions
# label_support_path = '/mnt/dgx-server/projects2023/dynunet_pipeline_label_merging/data/tasks/task3061_mindaomic/models/merged_model/fold0/label_merging/label_support.pt.npz'
# # reference label image
# label_path_ref = '/mnt/dgx-server/projects2023/dynunet_pipeline_label_merging/data/tasks/task3061_mindaomic/input/dataset/labelsTr/mindaomic_0002.nii.gz'
# # ground truth label image
# label_path_unmerged = '/mnt/dgx-server/projects2023/dynunet_pipeline_label_merging/data/tasks/task3061_mindaomic/results/inference/merged_model/fold0/mindaomic_0005.nii.gz'
# # read label to merged label mapping
# merged_labels_csv_path = '/mnt/dgx-server/projects2023/dynunet_pipeline_label_merging/data/tasks/task3061_mindaomic/models/merged_model/fold0/label_merging/merged_labels.csv'

# label support is needed to get the influence regions
label_support_path = '/mnt/dgx-server/projects2023/dynunet_pipeline_label_merging_from_label_support/data/tasks/task3061_mindaomic/models/merged_model/fold0/label_merging/label_support.pt.npz'
# reference label image
label_path_ref = '/mnt/dgx-server/projects2023/dynunet_pipeline_label_merging_from_label_support/data/tasks/task3061_mindaomic/input/dataset/labelsTr/mindaomic_0002.nii.gz'
#label_path_ref = '/mnt/dgx-server/projects2023/dynunet_pipeline_label_merging_from_label_support/data/tasks/task3061_mindaomic/results/inference/merged_model/fold0/mindaomic_0005.nii.gz'
# ground truth label image
#label_path_unmerged = '/mnt/dgx-server/projects2023/dynunet_pipeline_label_merging_from_label_support/data/tasks/task3061_mindaomic/input/dataset/labelsTr/mindaomic_0002.nii.gz'
label_path_unmerged = '/mnt/dgx-server/projects2023/dynunet_pipeline_label_merging_from_label_support/data/tasks/task3061_mindaomic/results/inference/merged_model/fold0/mindaomic_0005.nii.gz'
# read label to merged label mapping
merged_labels_csv_path = '/mnt/dgx-server/projects2023/dynunet_pipeline_label_merging_from_label_support/data/tasks/task3061_mindaomic/models/merged_model/fold0/label_merging/merged_labels.csv'


# # label support is needed to get the influence regions
# label_support_path = os.path.join(PROJ_ROOT, 'data', 'task2153_mind', 'output', 'label_support.pt.npz')
# # reference label image
# label_path_ref = os.path.join(PROJ_ROOT, 'data', 'task2153_mind', 'input', 'dataset', 'labelsTr', 'mind_000.nii.gz')
# # ground truth label image
# label_path_unmerged = os.path.join(PROJ_ROOT, 'data', 'task2153_mind', 'input', 'dataset', 'labelsTs', 'mind_038.nii.gz')
# label_path_unmerged = os.path.join(PROJ_ROOT, 'data', 'task2153_mind', 'input', 'dataset', 'labelsTr', 'mind_002.nii.gz')
# # read label to merged label mapping
# merged_labels_csv_path = os.path.join(PROJ_ROOT, 'data', 'task2153_mind', 'output', 'merged_labels.csv')


channel_to_label_mapping = pd.read_csv(merged_labels_csv_path, index_col='channel').to_dict()['merged_label']
merged_labels_df = pd.read_csv(merged_labels_csv_path, index_col='label')
label_to_merged_label_mapping = merged_labels_df['merged_label'].to_dict()


pad = 10

# perturb the ground truth label image with an affine transformation
transformed_label_merged, split_label_grtr, label_data_merged_mni, R = (
    get_network_output_from_registered_image(label_path_unmerged, pad))


label_nii_ref = nib.load(label_path_ref)
label_data_ref = label_nii_ref.get_fdata()

# pad the reference label image
label_data_ref = np.pad(label_data_ref, pad, mode='constant', )
label_data_ref_merged = map_labels_in_volume(label_data_ref, label_to_merged_label_mapping)

# get COMs
merged_labels = list(set(channel_to_label_mapping.values()))
coms_label1 = get_coms(label_data_ref_merged, merged_labels)

# get covariance for each label
covs_label1 = get_covs(label_data_ref_merged, merged_labels)


# get the com and covariance for the perturbed image
coms_label2 = get_coms(transformed_label_merged, merged_labels)
covs_label2 = get_covs(transformed_label_merged, merged_labels)


# define the weights for the COMs
coms_weights = torch.ones(coms_label1.shape[0], 1)*torch.arange(coms_label1.shape[0])[..., None]
# coms_weights = 1/com_stds_label1
# coms_weights = coms_weights/sum(coms_weights) * len(coms_weights)
coms_weights

# remove background label
coms_label1_no_bg = coms_label1[1:]
coms_label2_no_bg = coms_label2[1:]
covs_label1_no_bg = covs_label1[1:]
covs_label2_no_bg = covs_label2[1:]
coms_weights_no_bg = coms_weights[1:]

# if there are nan values in either com1, com2, cov1, or cov2, remove the corresponding label
nan_mask = (
        ~torch.any(torch.isnan(coms_label1_no_bg), dim=1)
        & ~torch.any(torch.isnan(coms_label2_no_bg), dim=1)
        & ~torch.any(torch.isnan(covs_label1_no_bg.reshape(len(covs_label1_no_bg), -1)), dim=1)
        & ~torch.any(torch.isnan(covs_label2_no_bg.reshape(len(covs_label2_no_bg), -1)), dim=1)
)
print(f"Remove {len(nan_mask) - sum(nan_mask).item()} labels with nan values")

coms_label1_matched = coms_label1_no_bg[nan_mask]
coms_label2_matched = coms_label2_no_bg[nan_mask]
covs_label1_matched = covs_label1_no_bg[nan_mask]
covs_label2_matched = covs_label2_no_bg[nan_mask]
coms_weights_matched = coms_weights_no_bg[nan_mask]

# define inputs for the optimization on the CPU
coms = coms_label1_matched.cpu().numpy()
coms2 = coms_label2_matched.cpu().numpy()
covs = covs_label1_matched.cpu().numpy()
covs2 = covs_label2_matched.cpu().numpy()
coms_weights = coms_weights_matched.cpu().numpy()

# run optimization
R_est = optimize_affine_labelreg(coms, coms2, covs, covs2, coms_weights)

# make homogeneous coordinates
R_est = np.concatenate([R_est, np.array([[0, 0, 0, 1]])], axis=0)

print(R)

# load the influence regions
influence_regions_path = ""
if not os.path.exists(influence_regions_path):

    fuzzy_prior_fudged = get_fuzzy_prior_fudged(label_support_path)

    label_to_channel_mapping = merged_labels_df['channel'].to_dict()

    influence_regions = get_influence_regions(fuzzy_prior_fudged, label_to_merged_label_mapping,
                                              label_to_channel_mapping)

    del fuzzy_prior_fudged
    # # save the influence regions
    # np.save(influence_regions_path, influence_regions)
else:
    influence_regions = np.load(influence_regions_path, allow_pickle=True).item()

# transform the influence regions
transformed_influence_regions = affine_transform_influence_regions(influence_regions, R_est, pad_val=pad)

# register the original label image to the transformed label image to check if the registration is correct
transformed_label_merged_R_est = affine_transform_image(label_data_merged_mni, torch.linalg.inv(torch.tensor(R_est)))

# split the transformed label using the transformed influence regions
split_label = split_merged_label(transformed_label_merged, transformed_influence_regions)

# plot results
plot_results([
    (transformed_label_merged, "transformed_label_merged"),
    (transformed_label_merged_R_est, "transformed_label_merged_R_est"),
    (transformed_label_merged.cpu().numpy() - transformed_label_merged_R_est, "diff"),
    (split_label, "split_label"),
    #(transformed_argmax_label_support_R_est, "transformed_argmax_label_support_R_est"),
]
)

# calculate dice
dice = get_dice(split_label_grtr.cpu().int(), split_label.cpu().int())
print(f'Dice: {dice}')

# plot the dice scores
plt.figure()
plt.plot(dice.values())
# y from 0 to 1
plt.ylim(0, 1.1)

plt.show()


# save the grtr and the split label
nib.save(nib.Nifti1Image(split_label.cpu().numpy(), label_nii_ref.affine), 'split_label.nii.gz')
nib.save(nib.Nifti1Image(split_label_grtr.cpu().numpy(), label_nii_ref.affine), 'split_label_grtr.nii.gz')