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

    def __init__(self, params, A, D):
        self.params = params
        self.A = A
        self.D = D

    def __call__(self, z, transpose=False):
        """
        Return z = b + alpha*D^gamma *z_l-1 + beta * D^gamma-1 A z^l-1
        """
        p = GraphLayer.params_transform(self.params)
        return (
            p[3] + p[0] * self.D ** p[2] @ z + p[1] * self.D ** (p[2] - 1) @ self.A @ z
        )

    def efficient_logdet_G_l(self):
        pass

    def get_G(self):
        p = GraphLayer.params_transform(self.params)
        G = p[0] * self.D ** p[2] + p[1] * self.D ** (p[2] - 1) * self.A
        return G

    @staticmethod
    def params_transform(params):
        alpha = params[0]  # jnp.exp(params[0])
        beta = params[1]  # alpha * jnp.tanh(params[1])
        gamma = jnp.exp(params[2])
        b = params[3]
        return jnp.array([alpha, beta, gamma, b])

    @staticmethod
    def params_transform_inverse(a_params):
        """
        Useful when initializing from desired params
        """
        # theta1 = jnp.log(a_params[0])
        # theta2 = jnp.log(-a_params[1] / a_params[0])
        theta3 = jnp.log(a_params[2])  # jnp.log(a_params[2] / (1 - a_params[2]))
        return jnp.array([a_params[0], a_params[1], theta3, a_params[3]])
