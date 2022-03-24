import jax.numpy as np
import numpy as onp
from kernels.abstract_kernel import Kernel
from kernels.standard_kernels import RQKernel, SEKernel, Matern52Kernel


class LocallyPeriodic(Kernel):
    """
    k2 of the Mona-Loa kernel see GPML RW5 page 120
    """
    def __init__(self):
        self.nParams = 3
        self.rawSEScale = 1.0
        self.rawPeriodicScale = 1.0
        self.rawNugget = 1.0
        super().__init__()

    def set_params(self, rawParams):
        self.rawSEScale = rawParams[0]
        self.rawPeriodicScale = rawParams[1]
        self.rawNugget = rawParams[2]

    def evaluate_kernel(self, x=None, x2=None, withNugget=True):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x2 is None:
            taus = np.linalg.norm((x[:, None, :] - x[None, :, :]) + 1e-16, axis=2)
            tau_diffs = np.pi * taus
            K = np.exp(-0.5 * taus ** 2 / self._constrain_param_positive(self.rawSEScale)
                       - 2 * (np.sin(tau_diffs) ** 2 / self._constrain_param_positive(self.rawPeriodicScale)))
            if withNugget:
                return K + self._constrain_param_positive(self.rawNugget) * np.eye(K.shape[0])
            else:
                return K
        else:
            # no nugget since this doesn't have a diagonal
            if x2.ndim == 1:
                x2 = x2.reshape(-1, 1)
            taus = np.linalg.norm((x[:, None, :] - x2[None, :, :]) + 1e-16, axis=2)
            tau_diffs = np.pi * taus
            K = np.exp(-0.5 * taus ** 2 / self._constrain_param_positive(self.rawSEScale)
                       - 2 * (np.sin(tau_diffs) ** 2 / self._constrain_param_positive(self.rawPeriodicScale)))
            return K


class MaunaLoaKernel(Kernel):
    """
    Rasmussen's Mauna Loa Kernel. Pages 120-121, GPML book.
    NOTE: We parameterize lengthscales as l while Rasmussemn uses l^2.
    If using their learned params, adjust appropriately.
    """

    def __init__(self):
        self.nParams = 11

        self.rawSEweight1 = 1.0
        self.rawSEScale1 = 1.0

        self.rawPeriodicweight2 = 1.0
        self.rawPeriodicScale2 = 1.0
        self.rawSEScale2 = 1.0

        self.rawRQWeight3 = 1.0
        self.rawScale3 = 1.0
        self.rawShape3 = 1.0

        self.rawNoiseWeight4 = 1.0
        self.rawSEScale4 = 1.0

        self.rawNugget = 1.0
        # self.donotwarpLP = donotwarpLP
        self.vectorized_mmle_params = None
        super().__init__()
        self.composite = True  # Flag indicating that this kernel is composed of several other kernels.

    def set_params(self, rawParams, set_mmle=False):
        self.rawSEweight1 = rawParams[0]
        self.rawSEScale1 = rawParams[1]
        self.rawPeriodicweight2 = rawParams[2]
        self.rawPeriodicScale2 = rawParams[3]
        self.rawSEScale2 = rawParams[4]
        self.rawRQWeight3 = rawParams[5]
        self.rawScale3 = rawParams[6]
        self.rawShape3 = rawParams[7]
        self.rawNoiseWeight4 = rawParams[8]
        self.rawSEScale4 = rawParams[9]
        self.rawNugget = rawParams[10]
        if set_mmle:
            self.vectorized_mmle_params = rawParams

    def evaluate_kernel(self, x=None, x2=None, warp_stuff=None, withNugget=True):
        """
        For composite kernels, the optional dict 'warp_stuff' contains the unwarped data,
        and a list of kernels whose inputs are to be warped.
        This is useful when only inputs to some of the kernels making up the composite
        kernel are to be warped.
        """
        unwarped_x = x  # if warp_stuff is empty, we assume that the kernel is being evaluated with un warped data
        donotwarp_list = [True] * 4
        kernel_list = []
        gram_matrix_list = []
        if warp_stuff:
            donotwarp_list = warp_stuff['donotwarplist']
            unwarped_x = warp_stuff['unwarped_x']

        self.k_se = SEKernel()
        self.k_se.set_params([self.rawSEScale1, None])  # None for no nugget
        kernel_list.append(self.k_se)

        self.k_lp = LocallyPeriodic()
        self.k_lp.set_params([self.rawSEScale2, self.rawPeriodicScale2, None])
        kernel_list.append(self.k_lp)

        self.k_rq = RQKernel()
        self.k_rq.set_params([self.rawScale3, self.rawShape3, None])
        kernel_list.append(self.k_rq)

        self.k_noise = SEKernel()
        self.k_noise.set_params([self.rawSEScale4, None])  # None for no nugget
        kernel_list.append(self.k_noise)

        if x2 is None:
            for k, donotwarp in zip(kernel_list, donotwarp_list):
                if donotwarp:
                    gram_matrix_list.append(k.evaluate_kernel(unwarped_x, withNugget=False))
                else:
                    gram_matrix_list.append(k.evaluate_kernel(x, withNugget=False))
            KSE, KLP, KRQ, Knoise = gram_matrix_list
            final_k = self._constrain_param_positive(self.rawSEweight1) * KSE + \
                      self._constrain_param_positive(self.rawPeriodicweight2) * KLP + \
                      self._constrain_param_positive(self.rawRQWeight3) * KRQ + \
                      self._constrain_param_positive(self.rawNoiseWeight4) * Knoise + \
                      self._constrain_param_positive(self.rawNugget) * np.eye(KSE.shape[0])
        else:
            unwarped_x2 = x2
            if warp_stuff:
                unwarped_x2 = warp_stuff['unwarped_x2']
            for k, donotwarp in zip(kernel_list, donotwarp_list):
                if donotwarp:
                    gram_matrix_list.append(k.evaluate_kernel(unwarped_x, unwarped_x2, withNugget=False))
                else:
                    gram_matrix_list.append(k.evaluate_kernel(x, x2, withNugget=False))
            # no nugget since this doesn't have a diagonal
            KSE, KLP, KRQ, Knoise = gram_matrix_list
            final_k = self._constrain_param_positive(self.rawSEweight1) * KSE + \
                      self._constrain_param_positive(self.rawPeriodicweight2) * KLP + \
                      self._constrain_param_positive(self.rawRQWeight3) * KRQ + \
                      self._constrain_param_positive(self.rawNoiseWeight4) * Knoise

        return final_k

    def convert_hyperparams(self, rawParams):
        real_params = onp.zeros(11)
        for i in range(11):
            if (i == 7):
                real_params[i] = self._constrain_param_positive(rawParams[i])
            else:
                real_params[i] = np.sqrt(self._constrain_param_positive(rawParams[i]))
        return real_params

