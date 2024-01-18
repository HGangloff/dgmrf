"""
Graph layer
"""

import equinox as eqx
import jax
import jax.numpy as jnp


class GraphLayer(eqx.Module):
    """
    Define one layer of a DGMRF graph parametrization
    """

    params: jax.Array
    A: jax.Array
    D: jax.Array
    log_det_method: str
    k_max: int
    precomputations: jax.Array

    def __init__(self, params, A, D, log_det_method, key=None):
        self.params = params
        self.A = A
        self.D = D
        self.log_det_method = log_det_method

        if self.log_det_method == "eigenvalues":
            # Precomputation of the eigenvalues for the logdet
            D_1A = 1 / self.D @ self.A
            try:
                # the eigenvalue computation is forced on CPU and the result goes
                # back to GPU
                cpu = jax.devices("cpu")[0]
                gpu = jax.devices("gpu")[0]
                eigen_D_1A = jnp.linalg.eigvals(jax.device_put(D_1A, cpu))
                self.precomputations = jax.device_put(eigen_D_1A, gpu)
            except RuntimeError:
                # no GPU found so the computation is directly done on CPU
                self.precomputations = jax.linalg.eigvals(D_1A)
        elif self.log_det_method == "power_series":
            # Precomputation of the Tr(\tilde{A}^K)=E[u.T@\tilde{A}@u]
            # (Hutchinson trace estimator)
            DAD = self.D ** (-0.5) @ self.A @ self.D ** (-0.5)
            self.k_max = 50
            self.precomputations = jnp.zeros((self.k_max - 1,))
            for k in range(1, self.k_max):
                if k > 1:
                    DAD = DAD @ DAD
                key, subkey = jax.random.split(key, 2)
                u = jax.random.normal(subkey, shape=(A.shape[0], 1))
                self.precomputations.at[k - 1].set((u.T @ DAD @ u).squeeze())

    def __call__(self, z, transpose=False):
        """
        Return z = b + alpha*D^gamma *z_l-1 + beta * D^gamma-1 A z^l-1
        """
        p = GraphLayer.params_transform(self.params)
        return (
            p[3] + p[0] * self.D ** p[2] @ z + p[1] * self.D ** (p[2] - 1) @ self.A @ z
        )

    def efficient_logdet_G_l(self):
        """
        Efficient computation of the determinant of a G_l. We currently
        implemented the eigenvalue method (Section 3.1.1 of Oskarsson 2022)
        """
        p = GraphLayer.params_transform(self.params)
        if self.log_det_method == "eigenvalue":
            return jnp.sum(
                p[2] * jnp.log(jnp.diag(self.D))
                + jnp.log(p[0] + p[1] * self.precomputations)
            )
        if self.log_det_method == "power_series":
            return (
                self.A.shape[0] * jnp.log(p[0])
                + jnp.sum(p[2] * jnp.log(jnp.diag(self.D)))
                + jnp.sum(
                    jnp.array(
                        [-1 / k * (-p[1] / p[0]) ** k for k in range(1, self.k_max)]
                    )
                    * self.precomputations
                )
            )

    def get_G(self):
        p = GraphLayer.params_transform(self.params)
        G = p[0] * self.D ** p[2] + p[1] * self.D ** (p[2] - 1) * self.A
        return G

    @staticmethod
    def params_transform(params):
        # alpha = params[0]  # jnp.exp(params[0])
        # beta = params[1]  # alpha * jnp.tanh(params[1])
        alpha = jnp.exp(params[0])
        beta = -alpha * jnp.tanh(params[1])
        gamma = jnp.exp(params[2])
        b = params[3]
        return jnp.array([alpha, beta, gamma, b])

    @staticmethod
    def params_transform_inverse(a_params):
        """
        Useful when initializing from desired params
        """
        theta1 = jnp.log(a_params[0])
        theta2 = jnp.log(a_params[1] / a_params[0])
        theta3 = jnp.log(a_params[2])  # jnp.log(a_params[2] / (1 - a_params[2]))
        return jnp.array([theta1, theta2, theta3, a_params[3]])
        # return jnp.array([a_params[0], a_params[1], theta3, a_params[3]])
