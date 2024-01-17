"""
Utility functions
"""

import jax.numpy as jnp


def get_adjacency_matrix_lattice(H, W, periodic_boundaries=False, weights=None):
    """
    Return the adjacency matrix for a H x W regular lattice with 4 neighbor system

    Parameters
    ----------
    H
        Integer. Height of the lattice
    W
        Integer. Width of the lattice
    periodic_boundaries
        Boolean. Whether to consider periodic boundaries. Default is False
    weights
        jnp.array with shape (4,) for the top, bot, left, right neighbors, respectively. This way
        we return a weighted adjacency matrix. Default is None. This is useful for constructing the
        G matrix in convolutional DGMRF for example
    """
    all_ = jnp.arange(H * W).reshape((H, W))
    if not periodic_boundaries:
        all_ = jnp.pad(all_, pad_width=1, mode="constant", constant_values=H * W + 1)
    if weights is None:
        weights = jnp.array([1.0, 1.0, 1.0, 1.0])

    A = jnp.zeros((H * W, H * W))

    A = A.at[
        jnp.arange(A.shape[0]), jnp.roll(all_, shift=1, axis=0)[1:-1, 1:-1].flatten()
    ].set(
        weights[0]
    )  # get all top neighbors
    A = A.at[
        jnp.arange(A.shape[0]), jnp.roll(all_, shift=-1, axis=0)[1:-1, 1:-1].flatten()
    ].set(
        weights[1]
    )  # get all bot neighbors
    A = A.at[
        jnp.arange(A.shape[0]), jnp.roll(all_, shift=1, axis=1)[1:-1, 1:-1].flatten()
    ].set(
        weights[2]
    )  # get all left neighbors
    A = A.at[
        jnp.arange(A.shape[0]), jnp.roll(all_, shift=-1, axis=1)[1:-1, 1:-1].flatten()
    ].set(
        weights[3]
    )  # get all right neighbors

    return A


# A = get_adjacency_matrix_lattice(5, 5)
# plt.imshow(A)
# plt.show()
