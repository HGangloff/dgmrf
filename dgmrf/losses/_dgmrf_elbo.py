"""
The ELBO for Deep Gaussian Markov Random Fields
"""
import jax
import jax.numpy as jnp
import equinox as eqx


def dgmrf_elbo(params, static, y, key, N, Nq, M):
    """
    Parameters
    ----------
    params
        a pytree of parameters to optimize on. Must contain a key `"dgmrf"` for
        the DGMRF parameters, two keys `"log_S_phi"` and `"nu_phi"` for the
        variational distribution parameters and a key `"log_sigma"` for the
        noise parameter
    static
        a pytree of static parameters (as defined by the eqx.partition()
        function). Must contain a key `"dgmrf"` for the static parameters of
        the DGMRF
    y
        A jax array of the observations
    key
        A jax random key
    N
        An integer. The total number of observations
    Nq
        An integer. The number of Monte-Carlo samples to approximate the
        expectation
    M
        An integer. The number of non-masked observations
    """

    dgmrf = eqx.combine(params["dgmrf"], static["dgmrf"])

    # Some reparametrizations to avoid numerical errors
    S_phi = jnp.exp(params["log_S_phi"])
    sigma = jnp.exp(params["log_sigma"])

    def scan_Nq(carry, _):
        key = carry[0]
        key, subkey = jax.random.split(key, 2)
        eps = jax.random.normal(subkey, (N,))
        xi = params["nu_phi"] + jnp.sqrt(S_phi) * eps

        g_xi = dgmrf(xi).flatten()
        res = g_xi.T @ g_xi + 1 / (sigma**2) * (y - xi).T @ (y - xi)

        return (key,), res

    _, accu_mcmc = jax.lax.scan(scan_Nq, (key,), jnp.arange(Nq))
    res_mcmc = jnp.mean(accu_mcmc)

    log_det_S_phi = jnp.sum(params["log_S_phi"])
    log_det_G_theta = dgmrf.log_det()
    log_sigma = params["log_sigma"]

    # ELBO divided by N as stated in the supp material
    elbo_val = (
        1 / N * (0.5 * log_det_S_phi - M * log_sigma + log_det_G_theta - 0.5 * res_mcmc)
    )
    # Note that we return -elbo
    # jax.debug.print("{p}", p=elbo_val)
    return -elbo_val
