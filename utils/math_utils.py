import jax
import jax.numpy as jnp

inv_transform = lambda y: y + jnp.log(jnp.maximum(1 - jnp.exp(-y), 1e-26))
squared_loss = lambda eta, a1, a2: jnp.mean((eta * a1 - a2) ** 2)


def smoothmin(x):
    scale = 10.  # smoothmin approximation of min improves with increasing scale (>0)
    return - jax.scipy.special.logsumexp(-1 * scale * x) / scale


def smoothmax(x):
    scale = 10.  # smoothmin approximation of min improves with increasing scale (>0)
    return jax.scipy.special.logsumexp(scale * x) / scale