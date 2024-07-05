"""
Graph layer
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from jaxtyping import Float, Int, Array, Key, Bool


class GraphLayer(eqx.Module):
    """
    Define one layer of a DGMRF graph parametrization
    """

    params: Array
    # NOTE that after eqx filterings, A and D still appears as learnable
    # parameters, hence they need to be declared as static
    A: Array = eqx.field(static=True)
    D: Array = eqx.field(static=True)
    log_det_method: str
    with_bias: Bool = eqx.field(static=True)
    k_max: Int = None
    precomputations: Array
    non_linear: Bool = eqx.field(static=True)

    def __init__(
        self, params, A, D, log_det_method, with_bias=True, non_linear=False, key=None
    ):
        self.params = params
        if not isinstance(A, BCOO):
            self.A = BCOO.fromdense(A)
        else:
            self.A = A
        self.D = D
        self.with_bias = with_bias
        self.log_det_method = log_det_method
        self.non_linear = non_linear

        if len(jnp.argwhere(D == 0)) > 0:
            raise ValueError(
                "There are isolated edges in the graph "
                "that we currently cannot handle. The formula below "
                "would propagate NaNs (indeed, 1/self.D with 0-degree nodes)"
            )

        cpu = jax.devices("cpu")[0]
        try:
            gpu = jax.devices("gpu")[0]
        except:
            gpu = cpu  # there is no GPU

        if self.log_det_method == "eigenvalues":
            # Precomputation of the eigenvalues for the logdet
            D_1A = jnp.diag(1 / self.D) @ self.A
            try:
                # the eigenvalue computation is forced on CPU and the result goes
                # back to GPU
                eigen_D_1A = jnp.linalg.eigvals(jax.device_put(D_1A, cpu))
                self.precomputations = jax.device_put(eigen_D_1A, gpu)
            except RuntimeError:
                # no GPU found so the computation is directly done on CPU
                self.precomputations = jnp.linalg.eigvals(D_1A)
            self.k_max = None
        elif self.log_det_method == "power_series":
            # Should be prefered for large graph. Moreover as above, the
            # computation is done on CPU, because RAM can get easily overflowed
            # even with sparse matrix format

            # Precomputation of the Tr(\tilde{A}^K)=E[u.T@\tilde{A}@u]
            # (Hutchinson trace estimator)
            with jax.default_device(cpu):
                # to avoid memory error for large graphs the instruction
                # DAD = jnp.diag(self.D ** (-0.5)) @ self.A @ jnp.diag(self.D ** (-0.5))
                # should become
                DAD = jax.device_put(
                    (self.D ** (-0.5))[:, None]
                    * ((self.D ** (-0.5))[:, None] * self.A),
                    cpu,
                )

                # the above line remains a sparse BCOO matrix when self.A is

                self.k_max = 50
                self.precomputations = jnp.zeros((self.k_max - 1,))
                for k in range(1, self.k_max):
                    if k > 1:
                        DAD = DAD @ DAD
                    key, subkey = jax.random.split(key, 2)
                    u = jax.random.normal(subkey, shape=(A.shape[0], 1))
                    self.precomputations.at[k - 1].set((u.T @ DAD @ u).squeeze())
            self.precomputations = jax.device_put(self.precomputations, gpu)
        else:
            raise ValueError(
                "log_det_method must be either eigenvalues or power_series"
            )

    def __call__(
        self, z, transpose=False, with_bias=True, with_h=False, with_non_linearity=True
    ):
        """
        Return z = b + alpha*D^gamma *z_l-1 + beta * D^gamma-1 A z^l-1

        Parameters
        ----------
        z
            Actually z_{l-1}, the input to the layer
        transpose
            boolean. Do we use the transpose of the kernel. Default is False
        with_bias
            boolean. Whether the bias is used. Default is True
        with_h
            boolean. Whether we return the non-activated result as second
            output. Default is False
        """
        non_linear = self.non_linear and with_non_linearity
        p = GraphLayer.params_transform(self.params, with_slope=non_linear)
        if transpose:
            Gz = p[0] * z * self.D ** p[2] + p[1] * (z @ self.A.T) * self.D ** (
                p[2] - 1
            )
        else:
            Gz = p[0] * self.D ** p[2] * z + p[1] * self.D ** (p[2] - 1) * (self.A @ z)
        if self.with_bias and with_bias:
            Gz += p[3]
        if not non_linear:
            activated_Gz = Gz
        else:
            activated_Gz = jax.nn.leaky_relu(Gz, negative_slope=p[4])
        if with_h:
            return activated_Gz, Gz
        return activated_Gz

    def mean_logdet_G(self):
        """
        Efficient computation of the determinant of a G_l. We currently
        implemented the eigenvalue method (Section 3.1.1 of Oskarsson 2022)
        """
        p = GraphLayer.params_transform(self.params)
        if self.log_det_method == "eigenvalues":
            return jnp.mean(
                p[2] * jnp.log(self.D)
                + jnp.log(jnp.abs(p[0] + p[1] * self.precomputations))
            )
        if self.log_det_method == "power_series":
            return (
                1
                / self.A.shape[0]
                * (
                    self.A.shape[0] * jnp.log(p[0])
                    + jnp.sum(p[2] * jnp.log(self.D))
                    + jnp.sum(
                        jnp.array(
                            [-1 / k * (-p[1] / p[0]) ** k for k in range(1, self.k_max)]
                        )
                        * self.precomputations
                    )
                )
            )
        raise ValueError("log_det_method must be either eigenvalues or power_series")

    def get_G(self):
        p = GraphLayer.params_transform(self.params)
        G = (
            p[0] * jnp.diag(self.D ** p[2])
            + p[1] * jnp.diag(self.D ** (p[2] - 1)) @ self.A
        )
        return G

    @staticmethod
    def params_transform(params, with_slope=False):
        # alpha = params[0]  # jnp.exp(params[0])
        # beta = params[1]  # alpha * jnp.tanh(params[1])
        alpha = jnp.exp(params[0])
        # beta = -alpha * jnp.exp(params[1])
        beta = alpha * jax.nn.tanh(params[1])
        # gamma = jnp.exp(params[2])
        gamma = jax.nn.sigmoid(params[2])
        # b = params[3]
        if with_slope:
            slope = jax.nn.softplus(params[4])
            return (alpha, beta, gamma, params[3], slope)
        else:
            return (alpha, beta, gamma, params[3], 0.0)

    @staticmethod
    def params_transform_inverse(a_params):
        """
        Useful when initializing from desired params
        """

        def inv_softplus(x):
            return jnp.log(jnp.exp(x) - 1)

        theta1 = jnp.log(a_params[0])
        # theta2 = jnp.log(-a_params[1] / a_params[0])
        theta2 = jnp.arctanh(a_params[1] / a_params[0])
        # theta3 = jnp.log(a_params[2])
        theta3 = jnp.log(a_params[2] / (1 - a_params[2]))
        return (theta1, theta2, theta3, a_params[3], inv_softplus(a_params[4]))
