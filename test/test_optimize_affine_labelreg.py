import unittest
import numpy as np
import torch

from labelmergeandsplit.labelreg_utils import optimize_affine_labelreg


class TestOptimizeAffineLabelreg(unittest.TestCase):
    def test_optimize_affine_labelreg(self):
        """
        Test the optimization function that optimizes the transformation matrix R to minimize the distance between two sets
        of coms and covs.
        """

        # R is ground truth transformation matrix
        R_grtr = np.array([
            [1.4, 0, 0, 2],
            [0, 0.8, 0, 1],
            [0, 0, 1, -5],
        ])

        R_grtr_4x4 = np.concatenate([R_grtr, np.array([[0, 0, 0, 1]])], axis=0)

        # coms1 and coms2 are the two sets of coms of shape (n_coms, 4)
        # 4 being the homogenous coordinates (x, y, z, 1)
        coms1 = np.array([
            [1, 1, 1, 1],
            [3, 4, 5, 1],
            [0, 0, 1, 1],
        ])

        coms2 = coms1

        # covs1 and covs2 are the two sets of covs of shape (n_coms, 3, 3)
        covs1 = np.array([
            [
                [1, 0, 0],
                [0, 2, 0],
                [0, 0, 3],
            ],
            [
                [1, 0, 0],
                [0, 3, 0],
                [0, 0, 4],
            ],
            [
                [1, 0, 0],
                [0, 3, 0],
                [0, 0, 4],
            ],
        ])

        covs2 = covs1

        # coms_weights is the weights for the coms distances of shape (n_coms)
        coms_weights = np.array([
            1,
            1,
            1
        ])

        # perturb the coms1 and covs1 with the inverse of the ground truth transformation matrix
        # the optimization function should be able to recover the ground truth transformation matrix to undo
        # the perturbation
        inv_R_grtr_4x4 = np.linalg.inv(R_grtr_4x4)
        coms1 = (inv_R_grtr_4x4 @ coms1[:, :, None]).squeeze()
        covs1 = inv_R_grtr_4x4[:3, :3] @ covs1 @ inv_R_grtr_4x4[:3, :3].T

        # map all arrays to float32
        R_grtr, coms1, coms2, covs1, covs2, coms_weights = (
            map(lambda x: x.astype(np.float32), (R_grtr, coms1, coms2, covs1, covs2, coms_weights)))

        R_est = optimize_affine_labelreg(coms1, coms2, covs1, covs2, coms_weights)

        print("\n\nThe ground truth transformation matrix is:"
              f"\n{R_grtr.reshape(3, 4)}")

        # assert the estimated transformation matrix is close to the ground truth transformation matrix
        np.testing.assert_allclose(R_est, R_grtr.reshape(3, 4), atol=1e-2)
