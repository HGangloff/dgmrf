import pytest
import jax
import jax.numpy as jnp
from dgmrf.models import DGMRF
from dgmrf.utils import get_adjacency_matrix_lattice


def test_equality():
    L = 1
    H = W = 40

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key, 2)

    dgmrf_conv = DGMRF(
        subkey,
        L,
        height_width=(H, W),
        init_params=[jnp.array([4.0, -1, -1, -1, -1, 0.0])],
    )

    dgmrf_graph = DGMRF(
        subkey,
        L,
        A_D=(get_adjacency_matrix_lattice(H, W), 4 * jnp.ones(H * W)),
        init_params=[jnp.array([1.0, -1.0, 1.0, 0.0])],
        log_det_method="eigenvalues",
    )

    key, subkey = jax.random.split(key, 2)
    graph_sample = dgmrf_graph.sample(subkey)
    conv_sample = dgmrf_conv.sample(subkey)

    assert jnp.allclose(graph_sample, conv_sample)
