import nibabel as nib
import numpy as np
np.set_printoptions(precision=5, suppress=True)
import os
import matplotlib.pyplot as plt
from labelmergeandsplit.merging_utils import load_label_support, map_labels_in_volume
from labelmergeandsplit.splitting_utils import get_fuzzy_prior_fudged, split_merged_label, get_influence_regions
import pandas as pd

import torch

pd.options.display.float_format = '{:20,.2f}'.format
from scipy.ndimage import affine_transform
import cupy
from cupyx.scipy.ndimage import affine_transform as affine_transform_cupy

import sys
sys.path.append('/home/aaron/Dropbox/KCL/Tools/CustomScripts')
from calculate_dice import get_dice

from scipy.linalg import sqrtm
from scipy.optimize import least_squares

from functools import partial


def coms_distance(mu1, mu2, cov1, cov2, coms_weights, ba=np):
    wd = (mu1 - mu2) * coms_weights
    return wd.flatten()


def moments_distance(mu1, mu2, cov1, cov2, coms_weights, ba=np):
    trace_cov1 = ba.einsum("...ii", cov1)
    trace_cov2 = ba.einsum("...ii", cov2)
    cross_term = - 2 * ba.sum(ba.real(ba.sqrt(ba.linalg.eigvals(cov1 @ cov2))), axis=-1)

    moments = trace_cov1 + trace_cov2 + cross_term
    return moments


def wasserstein_distance(mu1, mu2, cov1, cov2, coms_weights, ba=np):
    coms_d = coms_distance(mu1, mu2, cov1, cov2, coms_weights, ba=ba)
    moments_d = moments_distance(mu1, mu2, cov1, cov2, coms_weights, ba=ba)

    wd = ba.concatenate([coms_d.flatten(), moments_d.flatten()])
    return wd


def residual(R, coms1, coms2, covs1, covs2, coms_weights=None, sub_loss_func=None, ba=np):
    # make R compatible with coms homogenous coordinates
    if ba == np:
        R = np.concatenate([R, np.array([0, 0, 0, 1])], axis=0)
    elif ba == torch:
        R = torch.concatenate([R, torch.tensor([0, 0, 0, 1])], axis=0)
    else:
        raise ValueError("ba must be either np or torch")

    R = R.reshape(4, 4)

    # transform coms1
    coms1_transformed = (R @ coms1[:, :, None]).squeeze()

    # transform covs1
    convs1_transformed = R[:3, :3] @ covs1 @ R[:3, :3].T

    # coms_weights might need singular dimension appended
    if len(coms_weights.shape) == 1:
        coms_weights = coms_weights[..., None]

    wds = sub_loss_func(coms1_transformed[:, :3], coms2[:, :3], convs1_transformed, covs2, coms_weights, ba=ba)

    return wds


def get_jac(*args):
    """
    Get the jacobian of the residual function. This function is an input to the scipy least_squares function.
    The arguments args are passed as a tuple to the least_squares function and then are unpacked here.
    :param args: tuple of arguments passed to the least_squares function
    :return: the jacobian of the residual function
    """

    # the Jacobian is is calculated with torch.autograd.functional.jacobian, which requires a function that takes
    # a single tensor input, therefore, we need to create a partial function with the other arguments passed to
    # the residual function

    residual_partial = partial(residual,
                               coms1=torch.tensor(args[1]),
                               coms2=torch.tensor(args[2]),
                               covs1=torch.tensor(args[3]),
                               covs2=torch.tensor(args[4]),
                               coms_weights=torch.tensor(args[5]) if args[5] is not None else None,
                               sub_loss_func=args[6],
                               ba=torch,
                               )

    # calculate the Jacobian using the torch.autograd.functional.jacobian function and convert it to numpy for scipy
    # least_squares
    jac = torch.autograd.functional.jacobian(
        func=residual_partial,
        inputs=torch.tensor(args[0]).float(),
    ).cpu().numpy()

    return jac


def get_coms(label_data, labels):
    coms = torch.ones([len(labels), 4])  # homogeneous coordinates, Nx4
    for i, l in enumerate(labels):
        label_mask = label_data == l
        if not torch.any(label_mask):
            # fill with nan
            label_center = torch.tensor([np.nan, np.nan, np.nan])
        else:
            label_center = torch.mean(torch.argwhere(label_mask).float(), axis=0)
        coms[i, :3] = label_center
    return coms


def get_covs(label_data_ref_merged, merged_labels):
    covs = torch.zeros([len(merged_labels), 3, 3])
    for i, l in enumerate(merged_labels):
        label_mask = label_data_ref_merged == l
        if torch.any(label_mask):
            label_coords = torch.argwhere(label_mask).float()
            covs[i, :, :] = torch.cov(label_coords.T)
        else:
            covs[i, :, :] = torch.eye(3) * np.nan
    return covs


def get_coms_from_label_support(label_support, channel_to_label_mapping):
    merged_labels = list(set(channel_to_label_mapping.values()))

    channel_coms = []
    for m in merged_labels:
        summed_label = torch.zeros_like(label_support[0, :, :, :])

        for c, l in channel_to_label_mapping.items():
            if l == m:
                summed_label += label_support[c, :, :, :]

        com = torch.mean(torch.argwhere(summed_label).float(), axis=0)
        com = torch.cat([com, torch.tensor([1], device="cuda")])  # homogeneous coordinates
        channel_coms.append(com)

    coms_label = torch.stack(channel_coms, dim=0).to("cpu")
    return coms_label, merged_labels


def get_covs_from_label_support(label_support, channel_to_label_mapping):
    merged_labels = list(set(channel_to_label_mapping.values()))

    channel_covs = []
    for m in merged_labels:
        summed_label = torch.zeros_like(label_support[0, :, :, :])

        for c, l in channel_to_label_mapping.items():
            if l == m:
                summed_label += label_support[c, :, :, :]

        label_coords = torch.argwhere(summed_label).float()
        cov = torch.cov(label_coords.T)
        channel_covs.append(cov)

    return torch.stack(channel_covs, dim=0).to("cpu")


def match_coms(coms_label1, coms_label2):
    nan_mask1 = torch.isnan(coms_label1[:, 0])
    nan_mask2 = torch.isnan(coms_label2[:, 0])
    nan_mask = nan_mask1 | nan_mask2
    coms_label1 = coms_label1[~nan_mask,]
    coms_label2 = coms_label2[~nan_mask, :]

    return coms_label1, coms_label2


# transform covs
def transform_covs(covs, R):
    covs2 = R[:3, :3] @ covs @ R[:3, :3].T
    return covs2


def optimize_affine_labelreg(coms, coms2, covs, covs2, coms_weights):
    # Step 1: Optimization based on coms distance
    # initial guess
    R_init = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ])

    loss_fun = coms_distance

    lsq_res = least_squares(residual,
                            R_init.flatten(),
                            jac=get_jac,
                            args=(
                                coms,
                                coms2,
                                covs,
                                covs2,
                                coms_weights,
                                loss_fun
                            ),
                            x_scale='jac',
                            verbose=2, )

    R_est_3x4 = lsq_res.x.reshape(3, 4)

    print(f"estimated transformation matrix after coms distance optimization: \n{R_est_3x4}")

    # Step 2: Optimization based on wasserstein distance
    R_init = R_est_3x4


    loss_fun = wasserstein_distance

    lsq_res = least_squares(residual,
                            R_init.flatten(),
                            #jac=get_jac,
                            args=(
                                coms,
                                coms2,
                                covs,
                                covs2,
                                coms_weights,
                                loss_fun,
                            ),
                            #tr_solver='lsmr',  # 'exact' seems to lead to nans when the Jacobian is calculated with torch
                            x_scale='jac',
                            verbose=2, )

    R_est_3x4 = lsq_res.x.reshape(3, 4)

    print(f"estimated transformation matrix after wasserstein distance optimization: \n{R_est_3x4}")

    return R_est_3x4


def affine_transform_image(label_data, affine_matrix, mode='constant', order=0, backend=None):
    # register the label image to the transformed label image

    affine_matrix = affine_matrix.cpu().numpy() if isinstance(affine_matrix, torch.Tensor) else affine_matrix

    # make sure bottom row is close ot [0, 0, 0, 1]
    assert np.allclose(affine_matrix[3, :], [0, 0, 0, 1], atol=1e-5)
    # set bottom row to exactly [0, 0, 0, 1]
    affine_matrix[3, :] = np.array([0, 0, 0, 1])

    # apply the affine transformation
    if backend == "cupy":
        affine_matrix = cupy.array(affine_matrix)
        registered_label = affine_transform_cupy(label_data, affine_matrix, order=order, mode=mode)
    else:
        registered_label = affine_transform(label_data.cpu().numpy(), affine_matrix, order=order, mode=mode)

    return registered_label


def affine_transform_influence_regions(influence_regions, R, pad_val):
    transformed_influence_regions = {}

    # make sure R is in homogenous coordinates
    R = R.cpu().numpy() if isinstance(R, torch.Tensor) else R
    if R.shape[0] == 3:
        R = np.concatenate([R, np.array([[0, 0, 0, 1]])], axis=0)

    for idx in influence_regions.keys():
        ir = influence_regions[idx]
        print(f"Registering influence region {idx}")
        # pad the influence region for the transform
        ir = np.pad(ir.cpu().numpy(), pad_val, mode='edge', )

        ir = cupy.array(ir)

        # multilabel interpolation
        ir_out = cupy.zeros((ir.shape[0], ir.shape[1], ir.shape[2]))
        ir_inter_max = cupy.zeros((ir.shape[0], ir.shape[1], ir.shape[2]))
        for idxidx, l in enumerate(np.unique(ir)):
            ir_inter = affine_transform_image((ir == l).astype(float),
                                              torch.linalg.inv(torch.tensor(R)),
                                              mode='nearest',
                                              order=1,
                                              backend="cupy")

            ir_out = cupy.where(cupy.array(ir_inter > ir_inter_max), l, ir_out)
            ir_inter_max = cupy.maximum(ir_inter_max, ir_inter)

        transformed_influence_regions[idx] = torch.tensor(cupy.asnumpy(ir_out), device="cuda").int()

    return transformed_influence_regions