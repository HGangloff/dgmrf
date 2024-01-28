"""
Convolutional layer
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from dgmrf.utils._utils import get_adjacency_matrix_lattice


class ConvLayer(eqx.Module):
    """
    Define one convolution of a DGMRF parametrization
    """

    params: jax.Array
    H: int = eqx.field(static=True)
    W: int = eqx.field(static=True)
    with_bias: bool = eqx.field(static=True)
    non_linear: bool = eqx.field(static=True)

    def __init__(self, params, H, W, with_bias=True, non_linear=False):
        """
        Parameters
        ----------
        params
            XXX
        H
            The height if the regular lattice (image)
        W
            The width if the regular lattice (image)
        with_bias
            A boolean. Whether to add a bias after the convolution. Default is
            True
        non_linear
            A boolean. Whether we introduce non-linearities (Parametric ReLu
            with learnt parameters) between the convolutional layers. Default
            is False
        """
        self.params = params
        self.H = H
        self.W = W
        self.with_bias = with_bias
        self.non_linear = non_linear

    def __call__(self, z, transpose=False, with_bias=True):
        """
        Return z = G_lz_l-1 + b, i.e., apply one convolution
        """
        a = ConvLayer.params_transform(self.params)
        w = jnp.array([[0, a[2], 0], [a[1], a[0], a[3]], [0, a[4], 0]])
        if transpose:
            w = w.T
        #
        Gz = jax.scipy.signal.convolve2d(z.reshape((self.H, self.W)), w, mode="same")
        if self.with_bias and with_bias:
            Gz = (Gz + a[5]).flatten()
        if not self.non_linear:
            a[6] = 0.0  # if linear DGMRF force a[6] = 0 to have leaky_relu eq to relu
        # leaky_relu to be sure we maintain a bijection x<->z
        return jax.nn.leaky_relu(Gz, negative_slope=a[6])

    def mean_logdet_G(self):
        """
        Efficient computation of the determinant of a G_l (Proposition 2)
        of Siden 2020.
        """
        a = ConvLayer.params_transform_light(self.params)

        def scan_fun(_, ij):
            i, j = jnp.unravel_index(ij, (self.H, self.W))
            log_det_ij = jnp.log(
                jnp.abs(
                    a[0]
                    + 2 * jnp.sqrt(a[2]) * jnp.cos(jnp.pi * i / (self.H + 1))
                    + 2 * jnp.sqrt(a[1]) * jnp.cos(jnp.pi * j / (self.W + 1))
                )
            )
            return (), log_det_ij

        _, accu_log_det_ij = jax.lax.scan(scan_fun, (), jnp.arange(self.H * self.W))

        return jnp.mean(accu_log_det_ij)

    def get_G(self):
        a = ConvLayer.params_transform(self.params)
        G = get_adjacency_matrix_lattice(
            self.H, self.W, weights=jnp.array([a[2], a[4], a[1], a[3]])
        )
        G += a[0] * jnp.eye(self.H * self.W, self.H * self.W)
        return G

    @staticmethod
    def params_transform(params):
        a1 = jax.nn.softplus(params[0]) + jax.nn.softplus(params[1])
        sqrt_a2a4 = jax.nn.softplus(params[0]) * jax.nn.tanh(params[2]) / 2
        a4_a2 = jnp.exp(params[3])
        a2 = sqrt_a2a4 / jnp.sqrt(a4_a2)
        # NOTE, we lost the fact that a2 and a4 could be negative ?
        a4 = sqrt_a2a4 * jnp.sqrt(a4_a2)
        sqrt_a3a5 = jax.nn.softplus(params[1]) * jax.nn.tanh(params[4]) / 2
        a5_a3 = jnp.exp(params[5])
        a3 = sqrt_a3a5 / jnp.sqrt(a5_a3)
        a5 = sqrt_a3a5 * jnp.sqrt(a5_a3)
        # NOTE we choose to consider the neighboring values as negatives !
        # To be able to have the equivalency between Conv and Graph layer
        a7 = jnp.softplus(params[7])
        # NOTE: no constraint on a6 which is the bias and positivity constraint
        # on a7 which is the parameter of the Leaky Relu
        return jnp.array([a1, -a2, -a3, -a4, -a5, params[6], a7])

    @staticmethod
    def params_transform_light(params):
        a1 = jax.nn.softplus(params[0]) + jax.nn.softplus(params[1])
        a2a4 = (jax.nn.softplus(params[0]) * jax.nn.tanh(params[2]) / 2) ** 2
        a3a5 = (jax.nn.softplus(params[1]) * jax.nn.tanh(params[4]) / 2) ** 2
        return jnp.array([a1, a2a4, a3a5])

    @staticmethod
    def params_transform_inverse(a_params):
        """
        Useful when initializing from desired params
        """

        def inv_softplus(x):
            return jnp.log(jnp.exp(x) - 1)

        r2 = inv_softplus(2.0)  # arbitrary choice
        r1 = inv_softplus(a_params[0] - jax.nn.softplus(r2))
        r3 = jnp.arctanh(2 * jnp.sqrt(a_params[1] * a_params[3]) / jax.nn.softplus(r1))
        r4 = jnp.log(a_params[3] / a_params[1])
        r5 = jnp.arctanh(2 * jnp.sqrt(a_params[2] * a_params[4]) / jax.nn.softplus(r2))
        r6 = jnp.log(a_params[4] / a_params[2])
        return jnp.array(
            [r1, r2, r3, r4, r5, r6, a_params[5], inv_softplus(a_params[6])]
        )
