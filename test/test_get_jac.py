import unittest
import numpy as np

from labelmergeandsplit.labelreg_utils import get_jac, wasserstein_distance


class TestGetJac(unittest.TestCase):
    def test_get_jac(self):
        """
        Test the function that calculates the jacobian of the residual function using torch.autograd.functional.jacobian
        This gets the jacobian of the residual function by analytical differentiation, which is more accurate than
        the default numerical differentiation used by scipy least_squares.
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
        coms_weights = np.array([1, 1])[..., None]

        # loss between transformed coms1/covs1 and original coms2/covs2
        loss_fun = wasserstein_distance

        R, coms1, coms2, covs1, covs2, coms_weights = (
            map(lambda x: x.astype(np.float32), (R, coms1, coms2, covs1, covs2, coms_weights)))

        args = (R, coms1, coms2, covs1, covs2, coms_weights, loss_fun)

        # get the jacobian of the residual function
        jac = get_jac(*args)

        # the jacobian has shape (n_labels * 4, 12), where factor 4 comes from the fact that each label contributes
        # 4 residuals, one for each of the three spatial directions of the coms and one for the covs. 12 comes from the
        # fact that R has 12 parameters to be optimized.

        print(repr(jac))
        self.assertTrue(np.allclose(
            jac,
            np.array(
                [[1., 1., 1., 1., 0., 0.,
                  0., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 1., 1.,
                  1., 1., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0., 0.,
                  0., 0., 1., 1., 1., 1.],
                 [3., 4., 5., 1., 0., 0.,
                  0., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 3., 4.,
                  5., 1., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0., 0.,
                  0., 0., 3., 4., 5., 1.],
                 [1.51103, -1.15866, -4.15571, 0., -0.85111, -0.6151,
                  0.30233, 0., -1.36021, -0.95711, -0.409, 0.],
                 [1.49726, -1.38039, -5.9013, 0., -0.73166, -0.56582,
                  0.29199, 0., -1.52639, -1.1614, -0.29953, 0.]],
            ),
            atol=1e-5
        )
        )
