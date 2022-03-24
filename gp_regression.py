import jax
import jax.numpy as np
import numpy as onp
import scipy
jax.config.update("jax_enable_x64", True)


class GPregression():
    def __init__(self, x, y, kobj):
        """ Gaussian Process regression
        :param x: features / covariates
        :param y: responses
        :param kobj: an object of a class inheriting from Kernel
        """
        if x.ndim == 1:
            self.x = x.reshape(-1, 1)
        self.y = y
        self.n = self.x.shape[0]
        self.kernel_obj = kobj

    def neg_marginal_lik(self, params):
        self.kernel_obj.set_params(params)
        K = self.kernel_obj.evaluate_kernel(self.x)
        alpha = np.linalg.solve(K, self.y)
        return -(-0.5 * np.inner(self.y, alpha) - 0.5 * np.linalg.slogdet(K)[1]
                 - self.n / 2 * np.log(2 * np.pi))

    def fit(self, nRestarts=5):
        obj = lambda params: self.neg_marginal_lik(params)
        grad = jax.grad(obj)

        objVals = []
        paramVals = []
        for rr in range(nRestarts):
            init = onp.random.normal(size=self.kernel_obj.nParams)
            ret = scipy.optimize.minimize(fun=obj,
                                          jac=grad,
                                          x0=init)
            if ret.success:
                objVals.append(ret.fun)
                paramVals.append(ret.x)

        if len(objVals) == 0:
            return False
        objVals = np.array(objVals)
        paramVals = np.array(paramVals)
        bestIdx = np.argmin(objVals)
        self.kernel_obj.vectorized_mmle_params = paramVals[bestIdx]
        self.kernel_obj.set_params(paramVals[bestIdx])
        return True

    @staticmethod
    def predict(Ktrain, Ktesttrain, Ktest, y):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        L = np.linalg.cholesky(Ktrain)
        Linvy = np.linalg.solve(L, y)
        Linvk = np.linalg.solve(L, Ktesttrain.T)
        predictive_mean = np.dot(Linvk.T, Linvy)
        predictive_var = Ktest - np.dot(Linvk.T, Linvk)
        # from IPython import embed; embed()
        return predictive_mean.ravel(), np.diag(predictive_var)
