import numpy as onp
import matplotlib.pyplot as plt
from perturbations.input_warp import InputWarpPerturbation
from gp_regression import GPregression
from kernels.standard_kernels import Matern52Kernel, SEKernel
from perturbations.qi_checks import bootstrap
from perturbations.qi_checks import prior_draws, compute_matrix_diffs


# generate a dataset #
onp.random.seed(111)
n_train = 50
n_test = 500
n_perturb = 3
x = onp.linspace(-3, 3, n_train)
xtest = onp.sort(onp.random.randn(n_test) + 2.)
x_reg_grid = onp.linspace(x.min(), x.max(), 600)
f = onp.sin(x) + 2 * onp.cos(x ** 2)
y = f + 0.2 * onp.random.randn(n_train)
ftest = onp.sin(xtest) + 2 * onp.cos(xtest ** 2)
ytest = ftest + 0.2 * onp.random.randn(n_test)
x_star = onp.linspace(3.25, 3.75, n_perturb)
y_star = (onp.sin(x_star) + 2 * onp.cos(x_star ** 2)) * 0.01 - 2.


kobj = SEKernel() # Matern52Kernel() # choose from standard_kernels or compose your own
gpr = GPregression(x, y, kobj)
a = gpr.fit(nRestarts=5)
kobj = gpr.kernel_obj  # return fit kernel
# save_params
# joblib.dumps(kobj, 'data/...')

data = {'x': x, 'y': y, 'xtest': xtest, 'x_reg_grid': x_reg_grid}
perturbation_points = {'x': x_star, 'y': y_star - .5, 'order_statistic': 'None'}
perturb_kernel_obj = InputWarpPerturbation(kobj=kobj, epsilon=0.15, data=data,
                                           perturbation_points=perturbation_points)
perturb_kernel_obj.perturb_kernel(num_epochs=5001)
r = perturb_kernel_obj.predict_for_plots()
bootstrapped_params = bootstrap(x, y, num_bootstrap_samples=5, kobj=kobj)
plt.figure(figsize=(25, 4))
plt.plot(x, y, 'go', alpha=0.5, label="Training data")
# plt.plot(xtest, ytest, 'mo', alpha=0.5, label="Test data")
plt.plot(x_star, y_star - 0.5, 'r*', alpha=0.5, ms=10, label="Desired alternate predictions")
plt.plot(xtest, r['mu'], 'k--', label="k0")
plt.plot(xtest, r['pmu'], 'r--', label="k1")
plt.fill_between(x=xtest, y1=r['mu_ub'].ravel(), y2=r['mu_lb'].ravel(), color='k', alpha=0.2, )
plt.fill_between(x=xtest, y1=r['pmu_ub'].ravel(), y2=r['pmu_lb'].ravel(), color='r', alpha=0.2,)
plt.legend()

# qualitative interchangeablity checks
x_eval = x_reg_grid  # select locations to evaluate kernels on. Here we use the regularization grid.
kdict = perturb_kernel_obj.evaluate_original_alternate_kernel(x_eval)
k1 = kdict['k1']
k0 = kdict['k0']
fs = prior_draws(k1, k0, x_eval)
dists = compute_matrix_diffs(k0, k1)
fig, axs = plt.subplots(1, 2, figsize=(20, 4), sharex=True, sharey=True)
for i in onp.arange(10):
    axs[0].plot(x_eval, fs['f'][i], '-')
    axs[1].plot(x_eval, fs['fperturbed'][i], '-')
    axs[0].set_title("Samples from k0 (with nugget)")
    axs[1].set_title("Samples from k1 (with nugget)")
plt.show()

