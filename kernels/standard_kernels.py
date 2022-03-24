import jax.numpy as np
from kernels.abstract_kernel import Kernel


class SEKernel(Kernel):
    """
    Squared exponential kernel
    """
    def __init__(self):
        self.nParams = 2
        self.rawScale = 1.0
        self.rawNugget = 1.0
        super().__init__()

    def set_params(self, rawParams):
        self.rawScale = rawParams[0]
        self.rawNugget = rawParams[1]

    def evaluate_kernel(self, x=None, x2=None, withNugget=True):
        """
        Evaluate kernel between each row of x and X2, or between
        each pair of rows of x if X2 is None.
        """
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x2 is None:
            taus = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2)
            K = np.exp(-0.5 * taus / self._constrain_param_positive(self.rawScale))
            if withNugget:
                return K + self._constrain_param_positive(self.rawNugget) * np.eye(K.shape[0])
            else:
                return K
        else:
            # no nugget since this doesn't have a diagonal
            if x2.ndim == 1:
                x2 = x2.reshape(-1, 1)
            taus = np.sum((x[:, None, :] - x2[None, :, :]) ** 2, axis=2)
            K = np.exp(-0.5 * taus / self._constrain_param_positive(self.rawScale))
            return K


class Matern52Kernel(Kernel):
    """
    Matern 5/2 kernel
    """
    def __init__(self):
        self.nParams = 2
        self.rawScale = 1.0
        self.rawNugget = 1.0
        super().__init__()

    def set_params(self, rawParams):
        self.rawScale = rawParams[0]
        self.rawNugget = rawParams[1]

    def evaluate_kernel(self, x=None, x2=None, withNugget=True):
        """
        Evaluate kernel between each row of x and X2, or between
        each pair of rows of x if X2 is None.
        """
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x2 is None:
            taus = np.linalg.norm((x[:, None, :] - x[None, :, :]) + 1e-6, axis=2) # 1e-6 to avoid sqrt(0) in norm computation
            scale = self._constrain_param_positive(self.rawScale)
            K = ((1 + taus * np.sqrt(5) / scale + 5 * taus ** 2 / (3*scale ** 2)) *
                 np.exp(-taus * np.sqrt(5) / scale))
            if withNugget:
                return K + self._constrain_param_positive(self.rawNugget) * np.eye(K.shape[0])
            else:
                return K
        else:
            # no nugget since this doesn't have a diagonal
            if x2.ndim == 1:
                x2 = x2.reshape(-1, 1)
            taus = np.linalg.norm((x[:, None, :] - x2[None, :, :]) + 1e-6,
                                  axis=2)  # 1e-16 to avoid sqrt(0) in norm computation
            scale = self._constrain_param_positive(self.rawScale)
            K = ((1 + taus * np.sqrt(5) / scale + 5 * taus ** 2 / (3*scale ** 2)) *
                 np.exp(-taus * np.sqrt(5) / scale))
            return K


class RQKernel(Kernel):
    """
    Rational Quadratic Kernel
    """

    def __init__(self):
        self.nParams = 3
        self.rawScale = 1.0
        self.rawShape = 1.0
        self.rawNugget = 1.0
        super().__init__()

    def set_params(self, rawParams):
        self.rawScale = rawParams[0]
        self.rawShape = rawParams[1]
        self.rawNugget = rawParams[2]

    def evaluate_kernel(self, x=None, x2=None, withNugget=True):

        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x2 is None:
            taus = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2)
            K = (1 + 0.5 * taus / (self._constrain_param_positive(self.rawScale) *
                                        self._constrain_param_positive(self.rawShape))) ** \
                                        - self._constrain_param_positive(self.rawShape)
            if withNugget:
                return K + self._constrain_param_positive(self.rawNugget) * np.eye(K.shape[0])
            else:
                return K
        else:            # no nugget since this doesn't have a diagonal
            if x2.ndim == 1:
                x2 = x2.reshape(-1, 1)
            taus = np.sum((x[:, None, :] - x2[None, :, :]) ** 2, axis=2)
            K = (1 + 0.5 * taus / (self._constrain_param_positive(self.rawScale) *
                                   self._constrain_param_positive(self.rawShape))) ** \
                - self._constrain_param_positive(self.rawShape)
            return K