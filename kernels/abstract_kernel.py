import abc
import sys
import jax.numpy as np

# Ensure compatibility with Python 2/3
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class Kernel(ABC):
    """ Abstract kernel class. All implemented kernels will inherit this class
    """

    def __init__(self, *argv, **kwargs):
        """ Initialize Kernel
        """
        self.composite = False  # True if the kernel is a composition of simple kernels.

    @abc.abstractmethod
    def evaluate_kernel(self, *argv, **kwargs):
        """ Evaluate the kernel function on a set of points to produce a gram matrix
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_params(self, *argv, **kwargs):
        """ Helper function for setting kernel hyper-parameters
        """
        raise NotImplementedError

    def _constrain_param_positive(self, x):
        # Computes log(1 + e^x) in a way that is stabily jax'able;
        #  the point is that the gradient of np.log(1 + np.exp(x)) is unstable
        #  as is the gradient of np.log1p(np.exp(x))
        eps = 1e-6  # to avoid numerical
        return np.logaddexp(0, x) + eps