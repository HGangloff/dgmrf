"""
Cholesky sampling of a GRF; utility functions
"""

import jax
import jax.numpy as jnp


def euclidean_dist(x1, x2, y1, y2):
    return jnp.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def corr_exp(xy1, xy2, lx, ly, r):
    x1 = xy1 // lx
    y1 = xy1 % lx
    x2 = xy2 // lx
    y2 = xy2 % lx
    return jnp.exp(-euclidean_dist(x1, x2, y1, y2) / r)


def fill_diagonal(a, val):
    assert a.ndim >= 2
    i, j = jnp.diag_indices(min(a.shape[-2:]))
    return a.at[..., i, j].set(val)


def cholesky_sampling_gaussian_field(r, source_term, lx, ly):
    """
    Given the covariance matrix R ((lx*ly,lx*ly)) simulate a
    gaussian random field Y

    The vmapping enables to fill the big covariance matrix quickly
    """
    iterating = jnp.meshgrid(jnp.arange(lx * ly), jnp.arange(lx * ly))
    iterating = jnp.array([iterating[0].flatten(), iterating[1].flatten()])
    v_ = jax.jit(jax.vmap(corr_exp, (0, 0, None, None, None)))
    cov_mat = v_(iterating[0, :], iterating[1, :], lx, ly, r)
    cov_mat = cov_mat.reshape((lx * ly, lx * ly))
    cov_mat = fill_diagonal(cov_mat, 1.0)
    B = jnp.linalg.cholesky(cov_mat)
    Y = B @ source_term
    Y = Y.reshape((lx, ly))

    return Y
