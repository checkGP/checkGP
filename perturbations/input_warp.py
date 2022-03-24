import jax.numpy as jnp
import numpy as onp
import jax
from jax.experimental import stax, optimizers
from jax import jit
from jax import random
from gp_regression import GPregression
from utils.math_utils import squared_loss, smoothmin, smoothmax


class InputWarpPerturbation:
    """
    Input warping k(x, x') -> k(g(x), g(x')), where g(x) = x + perturbation_network(x, w).
    """

    def __init__(self, kobj, epsilon, data, perturbation_points, warp_stuff=None, perturbation_width=50):
        """
        kobj: Original kernel which will be warped. Must be instance of a class which implements a evaluate_kernel function
        epsilon: Defines epsilon ball around original kernel
        data: {'x': training covariates,
               'y': training responses,
               'xtest': test covariates,
               'x_reg_grid': grid of points to regularize the warp on}
        perturbation_points: {'x': np.array,
                              'y': np.array,
                              'order_statistic': "max" | "min" | None} A set of x, y pairs where x is the
                              perturbation point and y is the desired response at that point (or a desired orderstatistic
                              (max / min) of the responses at x). If order_statistic is max (min) then the max (min)
                              of the predicted responses at 'x' are used.
        warp_stuff (needed for composite kernels):
        {'donotwarplist': a boolean list indicating whether the kth base kernel's inputs should be warped;
                         warp inputs if False.
          'unwarped_x': unwarped covariates  (automatically populated)
        }

        perturbation_width: width of the dully connected two layer network.
        """
        self.kobj = kobj
        self.epsilon = epsilon
        self.init_perturb_network, self.perturbation_network = stax.serial(
            stax.Dense(perturbation_width), stax.Relu,
            stax.Dense(perturbation_width), stax.Relu,
            stax.Dense(1))
        self.x = data['x']
        self.xtest = data['xtest']
        self.x_reg_grid = data['x_reg_grid']
        self.y = data['y']
        self.warp_stuff = warp_stuff

        if self.x.ndim == 1:
            self.x = jnp.atleast_2d(self.x).T
        if self.xtest.ndim == 1:
            self.xtest = jnp.atleast_2d(self.xtest).T
        if self.x_reg_grid.ndim == 1:
            self.x_reg_grid = jnp.atleast_2d(self.x_reg_grid).T
        self.Xmean = jnp.mean(self.x, axis=0)
        self.Xstd = jnp.std(self.x, axis=0)
        self.perturbation_points = perturbation_points
        if self.perturbation_points['x'].ndim == 1:
            self.perturbation_points['x'] = jnp.atleast_2d(self.perturbation_points['x']).T
        if self.kobj.composite:
            self.warp_stuff['unwarped_x'] = onp.concatenate([self.x, self.perturbation_points['x']])
        self.standardized_X = (onp.concatenate([self.x, self.perturbation_points['x']]) - self.Xmean) / self.Xstd
        # num train / test / perturbation data points
        self.n_train = self.x.shape[0]
        self.n_test = self.xtest.shape[0]
        self.n_perturb = self.perturbation_points['x'].shape[0]
        self.n_perturb_responses = self.perturbation_points['y'].shape[0]

        self.nugget = jnp.eye(self.n_train)

        # perturbation network params
        self.perturbation_network_params = None

        # training Gram matrices
        self.K = None
        self.Kwarped = None
        # test Gram matrices
        self.Ktest = None
        self.Kwarpedtest = None
        # cross matrices
        self.Ktesttrain = None
        self.Kwarpedtesttrain = None
        r = self.evaluate_kernel(self.xtest, self.x)
        self.K = r['K']
        self.Ktest = r['Ktest']
        self.Ktesttrain = r['Ktesttrain']

        # sanity checks
        if self.n_perturb < self.n_perturb_responses:
            raise ValueError("number of perturbation responses can not be greater than "
                             "number of perturbation points")

    def evaluate_kernel(self, xtest, x, warped=False):
        if not warped:
            Kstar = self.kobj.evaluate_kernel(onp.concatenate([xtest, x]))
        else:
            # assert self.kobj.composite, "Partial warps only supported for composite kernels"
            if self.kobj.composite:
                ws = self.warp_stuff
                ws['unwarped_x'] = onp.concatenate([self.xtest, self.x])
                Kstar = self.kobj.evaluate_kernel(onp.concatenate([xtest, x]), warp_stuff=ws)
            else:
                Kstar = self.kobj.evaluate_kernel(onp.concatenate([xtest, x]))

        return {'Ktest': Kstar[:self.n_test, :self.n_test],
                'K': Kstar[self.n_test:, self.n_test:],
                'Ktesttrain': Kstar[:self.n_test, self.n_test:]
                }

    def warp_inputs(self, params, X):
        perturbation = self.perturbation_network(params, X)
        return X + perturbation

    def get_warped(self, params):
        # standardize x before feeding to network; destandardize gX after collecting from network.
        gX = self.warp_inputs(params, self.standardized_X)
        # from IPython import embed; embed()
        gX = self.Xmean + gX * self.Xstd
        if self.kobj.composite:
            Kwarped = self.kobj.evaluate_kernel(x=gX, warp_stuff=self.warp_stuff)
        else:
            Kwarped = self.kobj.evaluate_kernel(x=gX)
        return Kwarped, gX

    def _massage_warped_preds(self, warped_preds):
        if self.n_perturb > self.n_perturb_responses:
            # we want some order statistic of the predictions
            if self.perturbation_points['order_statistic'] == 'max':
                return smoothmax(warped_preds)
            elif self.perturbation_points['order_statistic'] == 'min':
                return smoothmin(warped_preds)
            elif not self.perturbation_points['order_statistic']:
                return warped_preds
            else:
                raise ValueError("Currently order_statistic can only take values 'max' xor 'min' xor None ")
        else:
            return warped_preds

    def posterior_predictive_loss(self, params):
        """
        :params: perturbation network parameters
        """
        Kwarped, gX = self.get_warped(params)
        K = Kwarped[:-self.n_perturb, :-self.n_perturb]
        Kperturbtrain = Kwarped[-self.n_perturb:, :-self.n_perturb]
        Kperturb = Kwarped[-self.n_perturb:, -self.n_perturb:]
        warped_preds, warped_pred_var = GPregression.predict(K, Kperturbtrain, Kperturb, self.y)
        warped_preds = self._massage_warped_preds(warped_preds)
        desired_preds = self.perturbation_points['y']
        # we will regularize on points selected from an uniform grid.
        gx_reg_grid = self.Xstd * self.warp_inputs(params, (self.x_reg_grid - self.Xmean) / self.Xstd) + self.Xmean
        return squared_loss(1., desired_preds, warped_preds) + (1. / self.epsilon) * jnp.mean((gx_reg_grid -
                                                                                               self.x_reg_grid) ** 2)

    def perturb_kernel(self, num_epochs=10001, seed=123):
        """
        learns and stores parameters of the perturbation network in self.perturbation_network_params
        """

        @jit
        def update(params, opt_state):
            """ Compute the gradient for a batch and update the parameters """
            value, grads = jax.value_and_grad(self.posterior_predictive_loss)(params)
            opt_state = opt_update(1, grads, opt_state)
            return get_params(opt_state), opt_state, value, grads

        rng = random.PRNGKey(seed)
        _, params = self.init_perturb_network(rng, (-1, 1))
        begin_lr, end_lr, decay_steps = 1e-2, 1e-4, 10000
        lr_schedule = optimizers.polynomial_decay(begin_lr, decay_steps, end_lr, power=1.0)
        opt_init, opt_update, get_params = optimizers.adam(lr_schedule)
        opt_state = opt_init(params)

        for epoch in onp.arange(num_epochs):
            params, opt_state, loss, grads = update(params, opt_state)
            if epoch % 1000 == 0:
                print("Epoch {0} | loss {1} ".format(epoch, loss))
        self.perturbation_network_params = params  # store final params

    def evaluate_perturbed_kernel(self):
        assert self.perturbation_network_params is not None
        params = self.perturbation_network_params
        # final warping
        gX = self.Xstd * self.warp_inputs(params, (self.x - self.Xmean) / self.Xstd) + self.Xmean
        # get test data
        gXtest = self.Xstd * self.warp_inputs(params, (self.xtest - self.Xmean) / self.Xstd) + self.Xmean
        # evaluate kernels on test points
        r = self.evaluate_kernel(gXtest, gX, warped=True)
        self.Kwarped = r['K']
        self.Kwarpedtest = r['Ktest']
        self.Kwarpedtesttrain = r['Ktesttrain']

    def predict_alternate_kernel(self):
        self.evaluate_perturbed_kernel()
        perturbed_mean, perturbed_var = GPregression.predict(self.Kwarped, self.Kwarpedtesttrain, self.Kwarpedtest,
                                                             self.y)
        return {'mu': perturbed_mean.ravel(), 'var': perturbed_var.ravel()}

    def predict_for_plots(self):
        alt_r = self.predict_alternate_kernel()
        perturbed_mean, perturbed_var = alt_r['mu'], alt_r['var']
        original_mean, original_var = GPregression.predict(self.K, self.Ktesttrain, self.Ktest, self.y)
        return {'mu': original_mean.ravel(), 'mu_ub': original_mean.ravel() + 3 * onp.sqrt(original_var),
                'mu_lb': original_mean.ravel() - 3 * onp.sqrt(original_var),
                'pmu': perturbed_mean.ravel(), 'pmu_ub': perturbed_mean.ravel() + 3 * onp.sqrt(perturbed_var),
                'pmu_lb': perturbed_mean.ravel() - 3 * onp.sqrt(perturbed_var)}

    def evaluate_original_alternate_kernel(self, x_eval):
        assert self.perturbation_network_params is not None
        if x_eval.ndim == 1:
            x_eval = jnp.atleast_2d(x_eval).T
        k0 = self.kobj.evaluate_kernel(x_eval)
        params = self.perturbation_network_params
        # final warping
        gx_eval = self.Xstd * self.warp_inputs(params, (x_eval - self.Xmean) / self.Xstd) + self.Xmean
        if self.kobj.composite:
            ws = self.warp_stuff
            ws['unwarped_x'] = x_eval
            k1 = self.kobj.evaluate_kernel(gx_eval, warp_stuff=ws)
        else:
            k1 = self.kobj.evaluate_kernel(gx_eval)
        return {'k0': k0, 'k1': k1, 'x_eval': x_eval}

