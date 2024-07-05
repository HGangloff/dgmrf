"""
Define the variational distributions one can use during ELBO maximization
"""

from abc import abstractmethod
import jax
import jax.numpy as jnp
import equinox as eqx

from dgmrf.models._dgmrf import DGMRF


class VariationalDistribution(eqx.Module):
    """
    Abstract variational distribution class
    """

    N: int

    def __init__(self, N):
        self.N = N

    @abstractmethod
    def get_variational_params(self):
        """
        Returns nu and S
        """

    def sample(self, key):
        """
        Sample with reparametrization trick

        Note this way to factorize code here while accessing child attributes
        with an abstract get method is called Template Method

        Note that S is a vector when using MeanField but a matrix when using
        FactorizedS, hence the condition
        """
        nu, S = self.get_variational_params()
        key, subkey = jax.random.split(key, 2)
        eps = jax.random.normal(subkey, (self.N,))
        if S.ndim == 1:
            return nu + S * eps
        return nu + S @ eps

    @abstractmethod
    def mean_log_det(self):
        """
        Returns the log determinant of the variational distribution
        """


class MeanField(VariationalDistribution):
    r"""
    The classical Gaussian mean-field variational distribution.
    q_{phi}(x)=N(x,\nu,SS^T) with \nu\in\mathbb{R}^N and log_S=diag(\log\sigma_1,
    ..., \log\sigma_N)
    """

    params: dict

    def __init__(self, N, nu_init=None, log_S_init=None, key=None):
        """
        One must provide a key if we do not provide an initial value for nu or
        S
        """
        super().__init__(N)
        self.params = {}
        if nu_init is None and key is not None:
            key, subkey = jax.random.split(key, 2)
            self.params["nu"] = jax.random.normal(subkey, (N,))
        else:
            self.params["nu"] = nu_init
        if log_S_init is None and key is not None:
            key, subkey = jax.random.split(key, 2)
            self.params["log_S"] = jax.random.normal(subkey, (N,)) * 0.5
        else:
            self.params["log_S"] = log_S_init

    def get_variational_params(self):
        return (self.params["nu"], jnp.exp(self.params["log_S"]))

    def mean_log_det(self):
        return jnp.mean(self.params["log_S"])


class FactorizedS(VariationalDistribution):
    r"""
    The variational distribution as proposed in Oskarsson 2022.
    q_{phi}(x)=N(x,\nu,SS^T) with \nu\ni\mathbb{R}^N and
    S=diag(\xi_1,...,\xi_N) G diag(\tau_1,...,\tau_N)
    where G is either a convolutional DGMRF layer or a graph DGMRF layer
    (or even a composition of such layers).
    """

    params: dict
    _dgmrf: eqx.Module

    def __init__(
        self,
        N,
        dgmrf_args,
        dgmrf_kwargs=None,
        nu_init=None,
        log_xi_init=None,
        log_tau_init=None,
        key=None,
    ):
        """
        One must provide a key if we do not provide an initial value

        dgmrf_args (resp. dgmrf_kwargs) represents all the positional parameters
        (resp. keyword parameters) that will be used
        to instanciate the dgmrf attribute from which we will only use the
        __call__ function to compute G, get_G_composition function
        and the log_det function
        """
        super().__init__(N)
        self.params = {}
        if nu_init is None and key is not None:
            key, subkey = jax.random.split(key, 2)
            self.params["nu"] = jax.random.normal(subkey, (N,))
        else:
            self.params["nu"] = nu_init
        if log_xi_init is None and key is not None:
            key, subkey = jax.random.split(key, 2)
            self.params["log_xi"] = jax.random.normal(subkey, (N,)) * 0.5
        else:
            self.params["log_xi"] = log_xi_init
        if log_tau_init is None and key is not None:
            key, subkey = jax.random.split(key, 2)
            self.params["log_tau"] = jax.random.normal(subkey, (N,)) * 0.5
        else:
            self.params["tau"] = log_tau_init

        if dgmrf_kwargs is None:
            dgmrf_kwargs = {}
        self._dgmrf = DGMRF(*dgmrf_args, **dgmrf_kwargs)

    def get_S(self):
        r"""
        S=diag(\xi_1,...,\xi_N) G diag(\tau_1,...,\tau_N)
        """
        # return (jnp.exp(self.params["log_xi"][:, None]) *
        #        (jnp.exp(self.params["log_tau"][:, None]) *
        #            self._dgmrf.get_G_composition()))
        return jnp.einsum(
            "i, ik, k -> ik",
            jnp.exp(self.params["log_xi"]),
            self._dgmrf.get_G_composition(),
            jnp.exp(self.params["log_tau"]),
        )

    def get_variational_params(self):
        return (self.params["nu"], self.get_S())

    def mean_log_det(self):
        return (
            self._dgmrf.mean_logdet()
            + jnp.mean(self.params["log_xi"])
            + jnp.mean(self.params["log_tau"])
        )
