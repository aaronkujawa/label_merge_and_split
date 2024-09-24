import unittest
import numpy as np
import torch

from labelmergeandsplit.labelreg_utils import residual, wasserstein_distance


class TestResidual(unittest.TestCase):
    def test_residual(self):
        """
        Test the residual function that calculates the residual between two sets of coms and covs, after applying a
        transformation matrix R to the first set of coms and covs.
        This residual can be used to optimize the transformation matrix R to minimize the distance between the two sets
        of coms and covs.
        """

        # R is the current transformation matrix
        R = np.array([
            [2, 0, -1, -1],
            [-1, 1, 0, 0],
            [0, 0, 1, 0],
        ]).flatten()

        # coms1 and coms2 are the two sets of coms of shape (n_coms, 4)
        # 4 being the homogenous coordinates (x, y, z, 1)
        coms1 = np.array([
            [1, 1, 1, 1],
            [3, 4, 5, 1],
        ])

        coms2 = np.array([
            [1, 1, 1, 1],
            [2, 2, 2, 1],
        ])

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
        ])

        covs2 = np.array([
            [
                [2, 0, 0],
                [0, 3, 0],
                [0, 0, 4],
            ],
            [
                [2, 0, 0],
                [0, 4, 0],
                [0, 0, 5],
            ],
        ])

        # coms_weights is the weights for the coms distances of shape (n_coms)
        coms_weights = np.array([1, 1])

        # loss between transformed coms1/covs1 and original coms2/covs2
        loss_fun = wasserstein_distance

        res = residual(R, coms1, coms2, covs1, covs2, coms_weights, sub_loss_func=loss_fun, ba=np)

        print(res)
        # the residuals coms1_x - coms2_x, coms1_y - coms2_y, coms1_z - coms2_z for each pair of coms, and the covs
        # residuals (one for each pair of covs) are concatenated into a single output array
        expected_res = [-1., -1., 0., -2., -1., 3., 3.00479, 3.76214]
        self.assertTrue(np.allclose(res, np.array(expected_res)))


        # test with torch (required to be able to calculate the jacobian with torch.autograd.functional.jacobian)
        R, coms1, coms2, covs1, covs2, coms_weights = map(torch.tensor, [R, coms1, coms2, covs1, covs2, coms_weights])
        R, coms1, coms2, covs1, covs2, coms_weights = map(lambda x: x.float(), [R, coms1, coms2, covs1, covs2, coms_weights])

        res = residual(R, coms1, coms2, covs1, covs2, coms_weights, sub_loss_func=loss_fun, ba=torch)

        print(res)
        self.assertTrue(torch.allclose(res, torch.tensor(expected_res)))
