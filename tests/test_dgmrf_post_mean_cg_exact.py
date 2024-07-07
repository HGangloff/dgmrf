import pytest
import os

os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
from dgmrf.models import DGMRF
from dgmrf.utils import get_adjacency_matrix_lattice


def test_eqality_mu_post_algo_convolutional():
    """
    We test that we get approximately the same result for the posterior mean
    estimation either with exact method or approximate (conjugate gradient)
    method. We test on a nonsense example.
    We use a convolutional DGMRF
    """
    L = 1
    H = W = 40

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key, 2)

    dgmrf = DGMRF(
        subkey,
        L,
        height_width=(H, W),
        init_params=[jnp.array([4.0, -1, -1, -1, -1, 0.0])],
    )

    key, subkey = jax.random.split(key, 2)
    # random observations
    tmp = dgmrf(jnp.zeros((H * W,)) + jax.random.normal(subkey, (H * W,)))
    # random mask
    tmp_mask = jnp.zeros((H, W))
    tmp_mask = tmp_mask.at[10:20, 10:20].set(1).flatten()

    mu_post_exact = dgmrf.get_post_mu(
        tmp, jnp.array([0.01]), mu0=tmp, mask=tmp_mask, method="exact"
    )
    mu_post_cg = dgmrf.get_post_mu(
        tmp, jnp.array([0.01]), mu0=tmp, mask=tmp_mask, method="cg"
    )

    assert (
        jnp.sum(
            jnp.round(mu_post_exact, decimals=2) == jnp.round(mu_post_cg, decimals=2)
        )
        == 1599
    )


def test_eqality_mu_post_algo_graph():
    """
    We test that we get approximately the same result for the posterior mean
    estimation either with exact method or approximate (conjugate gradient)
    method. We test on a nonsense example.
    We use a graph DGMRF
    """
    L = 1
    H = W = 40

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key, 2)

    dgmrf = DGMRF(
        subkey,
        L,
        A_D=(get_adjacency_matrix_lattice(H, W), 4 * jnp.ones(H * W)),
        init_params=[jnp.array([1.0, -1.0, 1.0, 0.0])],
        log_det_method="eigenvalues",
    )

    key, subkey = jax.random.split(key, 2)
    # random observations
    tmp = dgmrf(jnp.zeros((H * W,)) + jax.random.normal(subkey, (H * W,)))
    # random mask
    tmp_mask = jnp.zeros((H, W))
    tmp_mask = tmp_mask.at[10:20, 10:20].set(1).flatten()

    mu_post_exact = dgmrf.get_post_mu(
        tmp, jnp.array([0.01]), mu0=tmp, mask=tmp_mask, method="exact"
    )
    mu_post_cg = dgmrf.get_post_mu(
        tmp, jnp.array([0.01]), mu0=tmp, mask=tmp_mask, method="cg"
    )

    assert (
        jnp.sum(
            jnp.round(mu_post_exact, decimals=2) == jnp.round(mu_post_cg, decimals=2)
        )
        == 1599
    )
