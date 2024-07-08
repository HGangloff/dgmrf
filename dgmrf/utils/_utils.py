"""
Utility functions
"""

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
import numpy as np


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


def update_adjacency_matrix(adjacency_matrix, edge):
    i, j = edge
    adjacency_matrix = adjacency_matrix.at[i, j].set(1)
    adjacency_matrix = adjacency_matrix.at[j, i].set(1)
    return adjacency_matrix, None


def edge_list_to_adjacency_matrix(edge_list, num_nodes):
    """
    This function is optimized for GPU thanks to a JIT induced by the scan
    """
    edge_array = jnp.array(edge_list)
    adjacency_matrix = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int8)
    adjacency_matrix, _ = jax.lax.scan(
        update_adjacency_matrix, adjacency_matrix, edge_array
    )

    return adjacency_matrix


def get_N_y_D_A(filename):
    """
    Return the total number of nodes N, the target vector y, the degree matrix
    D and the adjacency matrix A. Note that the adjacency matrix is stored in
    the JAX BCOO format since it is sparse.
    """
    edges_mat = np.genfromtxt(
        f"./{filename}_edges.csv", delimiter=",", skip_header=1
    ).astype(np.int32)
    y = np.log(
        np.genfromtxt(f"./{filename}_target.csv", delimiter=",", skip_header=1) + 1e-6
    )[
        :, 1
    ]  # take log as in the original code (https://github.com/joeloskarsson/graph-dgmrf/blob/main/data_loading/wiki.py#L31)

    N = y.shape[0]

    ## Convert the edge list to adjacency matrix
    # if cpu_device is not None:
    #    with jax.default_device(cpu_device):
    #        A = edge_list_to_adjacency_matrix(edges_mat, y.shape[0])
    # else:
    A = edge_list_to_adjacency_matrix(edges_mat, y.shape[0])

    # Compute the diagonal of the degree matrix. It will not be in BCOO format
    D = jnp.sum(A, axis=1).astype(jnp.int32)

    print(f"Graph has {N} nodes with {jnp.sum(D)} edges")
    print(
        f"Adjacency matrix has {jnp.sum(D)/(A.shape[0]*A.shape[1]):.4f} % non zero entries"
    )

    return N, y, D, A


def get_y_with_mask_and_noise(y, mask_size, key, true_sigma_noise=None):
    """
    A random mask of size mask_size will be applied to y
    If a noise std is given, noise will be added to y
    """
    if true_sigma_noise is not None:
        key, subkey = jax.random.split(key, 2)
        y = y + jax.random.normal(subkey, y.shape) * true_sigma_noise

    mask = jnp.zeros_like(y)
    key, subkey = jax.random.split(key, 2)
    idx_unobserved = jax.random.choice(
        subkey, jnp.arange(y.shape[0]), shape=(mask_size,), replace=False
    )
    mask = mask.at[idx_unobserved].set(1)
    y_masked = jnp.where(mask == 0, y, 0)
    return y, y_masked, mask
