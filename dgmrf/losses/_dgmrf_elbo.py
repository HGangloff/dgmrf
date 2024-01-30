"""
The ELBO for Deep Gaussian Markov Random Fields
"""
import jax
import jax.numpy as jnp
import equinox as eqx


def dgmrf_elbo(params, static, y, key, N, Nq, mask=None):
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
    mask
        A jnp.array of 0 or 1 or True or False. Binary mask of masked observed variables. 1
        for masked, 0 for observed. Default is None
    """

    if mask is None:
        mask = jnp.zeros_like(y)
    if mask.dtype == bool:
        mask = mask.astype(int)

    dgmrf = eqx.combine(params["dgmrf"], static["dgmrf"])

    if dgmrf.non_linear:
        return non_linear_dgmrf_elbo(dgmrf, params, static, y, key, N, Nq, mask)
    return linear_dgmrf_elbo(dgmrf, params, static, y, key, N, Nq, mask)


def linear_dgmrf_elbo(dgmrf, params, static, y, key, N, Nq, mask=None):
    """
    ELBO for linear DGMRF
    """

    # Some reparametrizations to avoid numerical errors
    S_phi = jnp.exp(params["log_S_phi"])
    sigma = jnp.exp(params["log_sigma"])

    def scan_Nq(carry, _):
        key = carry[0]
        key, subkey = jax.random.split(key, 2)
        eps = jax.random.normal(subkey, (N,))
        xi = params["nu_phi"] + jnp.sqrt(S_phi) * eps

        g_xi = dgmrf(xi, with_bias=True)
        y_centered_masked = jnp.where(mask == 0, y - xi, 0)
        res = jnp.mean(g_xi**2) + 1 / (sigma**2) * jnp.mean(y_centered_masked**2)
        return (key,), res

    _, accu_mcmc = jax.lax.scan(scan_Nq, (key,), jnp.arange(Nq))
    res_mcmc = jnp.mean(accu_mcmc)

    log_det_S_phi = jnp.mean(params["log_S_phi"])
    log_det_G_theta = dgmrf.mean_logdet()
    log_sigma = params["log_sigma"]

    # ELBO divided by N as stated in the supp material
    elbo_val = (
        0.5 * log_det_S_phi
        - 1 / N * (N - jnp.sum(mask)) * log_sigma
        + log_det_G_theta
        - 0.5 * res_mcmc
    )
    # Note that we return -elbo
    # jax.debug.print("{p}", p=elbo_val)
    return -elbo_val


def non_linear_dgmrf_elbo(dgmrf, params, static, y, key, N, Nq, mask=None):
    """
    ELBO for non-linear DGMRF
    """

    def leaky_relu_derivative(x, negative_slope):
        """
        vmapping (vectorization) and derivative of the leaky relu on the first
        parameter
        """
        return jax.vmap(jax.grad(jax.nn.leaky_relu, argnums=0), in_axes=(0, None))(
            x, negative_slope
        )

    # Some reparametrizations to avoid numerical errors
    S_phi = jnp.exp(params["log_S_phi"])
    sigma = jnp.exp(params["log_sigma"])

    def scan_Nq(carry, _):
        key = carry[0]
        key, subkey = jax.random.split(key, 2)
        eps = jax.random.normal(subkey, (N,))
        xi = params["nu_phi"] + jnp.sqrt(S_phi) * eps

        z = xi
        log_det = 0
        log_non_linearities = 0
        for l in range(dgmrf.nb_layers):
            z, h = dgmrf.layers[l](z, with_h=True)
            log_det += dgmrf.layers[l].mean_logdet_G()
            derivatives = leaky_relu_derivative(
                h, jax.nn.softplus(dgmrf.layers[l].params[7])
            )
            # derivatives = jnp.where(derivatives < 1e-12, 1e-12, derivatives)
            log_non_linearities += jnp.mean(jnp.log(derivatives))

        y_centered_masked = jnp.where(mask == 0, y - xi, 0)
        res = (
            0.5 * jnp.mean(z**2)
            + 1 / (2 * sigma**2) * jnp.mean(y_centered_masked**2)
            + log_det
            + log_non_linearities
        )
        return (key,), res

    _, accu_mcmc = jax.lax.scan(scan_Nq, (key,), jnp.arange(Nq))
    res_mcmc = jnp.mean(accu_mcmc)

    log_det_S_phi = jnp.mean(params["log_S_phi"])
    log_sigma = params["log_sigma"]

    elbo_val = 0.5 * log_det_S_phi - 1 / N * (N - jnp.sum(mask)) * log_sigma - res_mcmc
    # Note that we return -elbo
    # jax.debug.print("{p}", p=elbo_val)
    return -elbo_val
