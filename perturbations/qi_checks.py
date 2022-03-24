import numpy as np
import scipy
from gp_regression import GPregression


def bootstrap(x=None, y=None, num_bootstrap_samples=200, kobj=None):
    """
    Performs bootstrap resampling
    """
    idx = np.arange(x.shape[0])
    x_std = np.std(x)
    y_std = np.std(y)
    bootstrapped_params = {'samples': []}
    # sample with replacement
    for n in np.arange(num_bootstrap_samples):
        s_idx = np.random.choice(idx, size=idx.shape)
        x_sample = x[s_idx] + 0.01 * x_std * np.random.randn(x.shape[0]) # jitter for duplicates
        y_sample = y[s_idx] + 0.01 * y_std * np.random.randn(y.shape[0]) # jitter for duplicates
        gpr = GPregression(x_sample, y_sample, kobj)
        gpr.fit(nRestarts=1)
        bootstrapped_params['samples'].append(gpr.kernel_obj.vectorized_mmle_params)
    return bootstrapped_params


def prior_draws(k0, k1, x_eval, num_draws=10):
    L = np.linalg.cholesky(k0 + 1e-16 * np.eye(k0.shape[0]))
    Lperturb = np.linalg.cholesky(k1 + 1e-16 * np.eye(k1.shape[0]))
    prior_draws = {'f': [], 'fperturbed': [], 'x': x_eval.tolist()}
    for i in np.arange(num_draws):
        eps = np.random.randn(x_eval.shape[0])
        # f = L @ eps + params['y']['mean']
        # fperturbed = Lperturb @ eps + params['y']['mean']
        f = L @ eps
        fperturbed = Lperturb @ eps
        prior_draws['f'].append([item.tolist() for item in f])
        prior_draws['fperturbed'].append([item.tolist() for item in fperturbed])
        # prior_draws['f'].append(L @ eps)
        # prior_draws['fperturbed'].append(Lperturb @ eps)
    return prior_draws


def compute_matrix_diffs(k0, k1):
    """
    Given two gram matrices k0 and k1, computes various "distances" between them.
    """
    return {
            'wasserstein': compute_wasserstein(k0, k1),
            'symmeterized_kl': compute_symmeterized_kl(k0, k1)
            }


def compute_wasserstein(k0, k1, print_warnings=False):
    """
    2-WD between zero mean Gaussians with covariances k0 and k1
    """

    # Scale the matrices down to a reasonable numerical range. Otherwise
    #  later numerical instabilities are made much worse (in particular, the
    #  minimum eigenvalue of K0_K1_K0 can be significantly negative)
    k0Scale = k0.max()
    k1Scale = k1.max()
    k0 /= k0Scale
    k1 /= k1Scale

    sqrtk0 = scipy.linalg.sqrtm(k0)
    first_term = sqrtk0 @ k1
    K0_K1_K0 = first_term @ sqrtk0

    # K0_K1_K0 is, in theory, a PSD matrix. It may have some negative eigenvalues
    #  due to numerical inaccuracy. Because we're taking the sqrt of the
    #  eigenvalues, this is a problem. Check that the minimum eigenvalue is
    #  positive or numerically 0. Then make them  all positive.
    eigvals = np.linalg.eigvalsh(K0_K1_K0)
    if eigvals.min() < 0:
        if not (np.allclose(eigvals.min(), 0)):
            from IPython import embed;
            np.set_printoptions(linewidth=80);
            embed()
        eigvals = np.maximum(eigvals, 0)

    wd = (np.trace(k0) * k0Scale
          + np.trace(k1) * k1Scale
          - 2 * np.sqrt(eigvals).sum() * np.sqrt(k0Scale) * np.sqrt(k1Scale)
          )

    k1 *= k1Scale
    k0 *= k0Scale
    return wd.tolist()


def compute_symmeterized_kl(k0, k1, eps=1e-8):
    # B/c of numerical instability, you might get some eigenvalues < 0.
    #   Round these up to some epsilon.
    vals1, vecs1 = np.linalg.eigh(k1)
    vals0, vecs0 = np.linalg.eigh(k0)
    vals1 = np.maximum(eps, vals1)
    vals0 = np.maximum(eps, vals0)

    # Computes trace10 = np.trace(np.linalg.solve(k0, k1))
    trace10 = np.trace(vecs0 @ np.diag(1 / vals0) @ vecs0.T @ vecs1 @ np.diag(vals1) @ vecs1.T)
    trace01 = np.trace(vecs1 @ np.diag(1 / vals1) @ vecs1.T @ vecs0 @ np.diag(vals0) @ vecs0.T)

    ret = 0.25 * (-2 * k0.shape[0]
                  + trace10
                  + trace01
                  )
    if ret < 0:
        from IPython import embed;
        np.set_printoptions(linewidth=80);
        embed()
    return ret
