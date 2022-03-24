import numpy as onp
import joblib
import matplotlib.pyplot as plt
from perturbations.input_warp import InputWarpPerturbation
from kernels.derived_kernels import MaunaLoaKernel
from utils.io_utils import read_mauna_loa
from gp_regression import GPregression
from perturbations.qi_checks import prior_draws, compute_matrix_diffs


def viz_fit():
    plt_x = xtest
    Ktrain = kobj.evaluate_kernel(x=x)
    Ktesttrain = kobj.evaluate_kernel(x=x, x2=plt_x).T
    Ktest = kobj.evaluate_kernel(x=plt_x)
    mu, var = GPregression.predict(Ktrain, Ktesttrain, Ktest, y)
    plt.plot(x, y, 'ko')
    plt.plot(plt_x, mu, 'r-')
    plt.fill_between(x=plt_x, y1=mu + 2 * onp.sqrt(var), y2=mu - 2 * onp.sqrt(var), color='r', alpha=0.2)
    plt.title("k0 predictions")


def viz_perturbed_fit():
    plt.figure(figsize=(25, 4))
    plt.plot(x, y, 'go', alpha=0.5, label="Training data")
    plt.plot(xtest, r['mu'], 'k--', label="k0")
    plt.plot(xtest, r['pmu'], 'r--', label="k1")
    plt.fill_between(x=xtest, y1=r['mu_ub'].ravel(), y2=r['mu_lb'].ravel(), color='k', alpha=0.2)
    plt.fill_between(x=xtest, y1=r['pmu_ub'].ravel(), y2=r['pmu_lb'].ravel(), color='r', alpha=0.2)
    plt.axhline(y=415.5, color='g', ls='--', lw=2.5, label="415 ppm threshold")
    plt.legend()


# load data
y, x, ytest, xtest, *_ = read_mauna_loa("./data/maunaloa.csv")
# define a grid of data points
x_reg_grid = onp.linspace(x.min(), xtest.max(), 600)
kobj = MaunaLoaKernel()
# load pre fit MMLE parameters from disk / optionally
mmle_params = joblib.load("./data/mauna_loa_mmle.pkl")
kobj.set_params(mmle_params['vectorized_mmle_params'])
# visualize fit
viz_fit()

# define perturbation parameters
epsilon = 0.15
data = {'x': x, 'y': y, 'xtest': xtest, 'x_reg_grid': x_reg_grid}
perturbation_points = {'x': onp.array([2019.1, 2019.6, 2020.1]), 'y': onp.array([415.5]), 'order_statistic': 'max'}
# warp_stuff is needed for composite kernels like the mauna-loa kernel.
donotwarplist = [False, True, False, False]  # do not warp inputs to the locally periodic component of the Mauna Loa kernel
warp_stuff = {'donotwarplist': donotwarplist, 'unwarped_x': None}
perturbed_obj = InputWarpPerturbation(kobj, epsilon, data, perturbation_points, warp_stuff, perturbation_width=50)
perturbed_obj.perturb_kernel(num_epochs=5001)
r = perturbed_obj.predict_for_plots()
viz_perturbed_fit()

# qualitative interchangeablity checks
x_eval = x_reg_grid  # select locations to evaluate kernels on. Here we use the regularization grid.
kdict = perturbed_obj.evaluate_original_alternate_kernel(x_eval)
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
# joblib.dump(kdict, "mauna-loa-kernels")