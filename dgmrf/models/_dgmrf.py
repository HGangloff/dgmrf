"""
DGMRF model
"""

import numpy as np
import equinox as eqx
import torch
from torch.distributions import MultivariateNormal
import jax
import jax.numpy as jnp

from dgmrf.layers._graph_layer import GraphLayer
from dgmrf.layers._conv_layer import ConvLayer


class DGMRF(eqx.Module):
    """
    Define a complete DGMRF parametrization

    We construct either a convolutional DGMRF or a graph DGMRF
    """

    key: jax.Array
    nb_layers: int = eqx.field(static=True)
    layers: list
    N: int
    non_linear: bool = eqx.field(static=True)

    def __init__(
        self,
        key,
        nb_layers,
        height_width=None,
        A_D=None,
        init_params=None,
        *args,
        **kwargs,
    ):
        """
        Parameters
        ----------
        key
            A JAX RNG key
        nb_layers
            An integer. The number of layers in the model
        height_width
            A tuple of integers (Height, Width) to create a convolutional DGMRF. Cannot be mutually equal to
            None with A_D. Default is None
        A_D
            A tuple of jnp.array to designate the (Adjacency matrix,
            **diagonal** of the Degree matrix) to create a graph DGMRF. Cannot
            be mutually equal to None with height_width. Defaut is None
        init_params
            Wether to use specific parameters for the DGMRF creation. If creating a convolutional DGMRF, init_params
            is list of jnp.array([a[0], a[1], a[2], a[3], a[4], b]) for each layer
            If creating a graph DMGRF, init_params is a list of jnp.array([alpha, beta, gamma, b]) for each layer
        args
            Diverse arguments that will be passed to the layer init function
        kwargs
            Diverse arguments that will be passed to the layer init function
        """
        if (height_width is None and A_D is None) or (
            height_width is not None and A_D is not None
        ):
            raise ValueError(
                "Either height and width or A and D should be specified, mutually exclusive"
            )

        self.key = key
        self.nb_layers = nb_layers
        self.layers = []
        for i in range(self.nb_layers):
            if height_width is not None:
                if init_params is not None:
                    self.layers.append(
                        ConvLayer(
                            ConvLayer.params_transform_inverse(init_params[i]),
                            height_width[0],
                            height_width[1],
                            *args,
                            **kwargs,
                        )
                    )
                else:
                    self.key, subkey = jax.random.split(self.key, 2)
                    self.layers.append(
                        ConvLayer(
                            # jax.random.uniform(subkey, (7,), minval=-1, maxval=1),
                            jax.random.normal(subkey, (8,)) * 0.1,
                            height_width[0],
                            height_width[1],
                            *args,
                            **kwargs,
                        )
                    )
                self.N = height_width[0] * height_width[1]
            if A_D is not None:
                if init_params is not None:
                    self.key, subkey = jax.random.split(self.key, 2)
                    self.layers.append(
                        GraphLayer(
                            GraphLayer.params_transform_inverse(init_params[i]),
                            A_D[0],
                            A_D[1],
                            *args,
                            **kwargs,
                            key=subkey,
                        )
                    )
                else:
                    self.key, subkey1, subkey2 = jax.random.split(self.key, 3)
                    self.layers.append(
                        GraphLayer(
                            # jax.random.uniform(subkey1, (4,), minval=-1, maxval=1),
                            jax.random.normal(subkey1, (4,)) * 0.1,
                            A_D[0],
                            A_D[1],
                            *args,
                            **kwargs,
                            key=subkey2,
                        )
                    )
                self.N = A_D[0].shape[0]

    def __call__(self, x, transpose=False, with_bias=True):
        """
        Return the composition of g = gLgL-1...g1(z0) with z0 = x if not
        transpose. Return g = g1...gL-1gL(z0) if transpose
        """
        z = x
        if transpose:
            for l in reversed(range(self.nb_layers)):
                z = self.layers[l](z, transpose=transpose, with_bias=with_bias)
        else:
            for l in range(self.nb_layers):
                z = self.layers[l](z, transpose=transpose, with_bias=with_bias)
        return z

    def mean_logdet(self):
        """
        Compute the log determinant
        """
        log_det = 0
        for l in range(self.nb_layers):
            log_det += self.layers[l].mean_logdet_G()
        return log_det

    def get_Q(self):
        """
        Get the precision matrix of the DGMRF using the formula Q = G^TG
        """
        if self.non_linear:
            raise ValueError(
                "Exact formula for Q is non available for " "non-linear DGMRF"
            )

        G = self.layers[0].get_G()
        for l in range(self.nb_layers - 1):
            G = self.layers[l + 1].get_G() @ G
        return G.T @ G

    def get_mu(self):
        """
        mu = G^-1*b = G^-1*g(0). We invert G with conjugate gradient
        """
        if self.non_linear:
            raise ValueError(
                "Exact formula for Q is non available for " "non-linear DGMRF"
            )

        def G(x):
            return self(x, with_bias=False)

        mu, _ = jax.scipy.sparse.linalg.cg(
            G, self(jnp.zeros((self.N,)), with_bias=True), x0=jnp.zeros((self.N,))
        )
        return mu

    def get_QTilde(self, x, log_sigma, mask=None):
        if self.non_linear:
            raise ValueError(
                "Exact formula for Q is non available for " "non-linear DGMRF"
            )

        if mask is None:
            mask = jnp.zeros_like(x)
        if mask.dtype == bool:
            mask = mask.astype(int)

        Gx = self(x, with_bias=False)
        GTGx = self(Gx, transpose=True, with_bias=False)
        return GTGx + jnp.where(mask == 0, 1 / (jnp.exp(log_sigma) ** 2) * x, 0)

    def get_post_mu(self, y, log_sigma, mu0=None, mask=None):
        """
        Compute the posterior mean with conjugate gradient as proposed in Siden
        2020, Oskarsson 2022 and Lippert 2023. We know that mu_post =
        (Q+1/sigma^2 I)^-1(-G^Tb+1/sigma^2 y) with b=g(0)
        We can give tilde{Q}=Q+1/sigma^2 I as a function which tells how
        to compute tilde{Q}x that's what we'll do to avoid explicitely
        constructing G

        Parameters
        ----------
        y
            The observations
        log_sigma
            The parameter for the noise level
        mu0
            The initial guess for the posterior mean. Default is None.
        mask
            A jnp.array of 0 or 1 or True or False. Binary mask of masked observed variables. 1
            for masked, 0 for observed. Default is None
        """
        if self.non_linear:
            raise ValueError(
                "Exact formula for Q is non available for " "non-linear DGMRF"
            )

        if mask is None:
            mask = jnp.zeros_like(y)
        if mask.dtype == bool:
            mask = mask.astype(int)

        b = self(jnp.zeros_like(mu0), with_bias=True)

        c = -self(b, transpose=True, with_bias=False) + jnp.where(
            mask == 0, 1 / (jnp.exp(log_sigma) ** 2) * y, 0
        )

        return jax.scipy.sparse.linalg.cg(
            lambda x: self.get_QTilde(x, log_sigma, mask),
            c,
            mu0,  # maxiter
        )[0]

    def sample(self, key):
        """
        Sample from the DGMRF

        Parameters
        ----------
        key
            A JAX random key
        """
        # from https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
        torch.manual_seed(key[0])
        torch.cuda.manual_seed(key[0])
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        mvn = MultivariateNormal(
            loc=torch.from_numpy(np.array(self.get_mu())),
            precision_matrix=torch.from_numpy(np.array(self.get_Q())),
        )
        return mvn.sample().numpy()

    def posterior_samples(self, nb_samples, y, log_sigma, key, x0=None):
        """
        Perform posterior sample with perturbation as described in Siden 2020
        for example. The approach has been proposed in Papandreou and Yuille
        2010

        Parameters
        ----------
        nb_samples
            The number of posterior samples to draw
        y
            The observations
        log_sigma
            The parameter for the noise level
        key
            A JAX random key
        x0
            An initial get for the solution
        """
        if self.non_linear:
            raise ValueError(
                "Exact formula for Q is non available for " "non-linear DGMRF"
            )

        b = self(jnp.zeros_like(y), with_bias=True)

        def get_one_posterior_sample(carry, _):
            (key,) = carry
            key, subkey1, subkey2 = jax.random.split(key, 3)
            u1 = jax.random.normal(subkey1, shape=b.shape)
            u2 = jax.random.normal(subkey2, shape=y.shape)
            c_perturbed = -self((u1 - b), transpose=True, with_bias=False) + 1 / (
                jnp.exp(log_sigma) ** 2
            ) * (y + jnp.exp(log_sigma) * u2)
            xpost_CG, _ = jax.scipy.sparse.linalg.cg(
                lambda x: self.get_QTilde(x, log_sigma), c_perturbed, x0
            )
            return (key,), xpost_CG

        key, subkey = jax.random.split(key, 2)
        _, x_post_samples = jax.lax.scan(
            get_one_posterior_sample, (subkey,), jnp.arange(nb_samples)
        )
        return x_post_samples

    def rbmc_variance(self, x_post_samples, log_sigma):
        """
        Get the Rao-Blackwellized Monte Carlo estimation for the variance as
        done in Siden 2020 (introduced in Siden 2018).
        We use a JVP-like way to get $G^TG$ for real. We know that we are
        able to compute the matrix vector product $G^TG(x)$. Each time we
        perform such a computation with $x$ being $0$ everywhere except
        at one place, we reveal one column of $G^TG$.
        So we do so repeatedly with a vmap.

        Parameters
        ----------
        x_post_samples
            A jnp.array with a list of posterior samples on the first axis.
            Those are used to get our estimation
        log_sigma
            The parameter for the noise level
        """
        if self.non_linear:
            raise ValueError(
                "Exact formula for Q is non available for " "non-linear DGMRF"
            )

        x_post_samples_demeaned = x_post_samples - jnp.mean(
            x_post_samples, axis=0, keepdims=True
        )
        v_QTilde = jax.vmap(lambda x: self.get_QTilde(x, log_sigma))
        diag_QTilde = jnp.diag(v_QTilde(jnp.eye(self.N)))
        var_x_post_samples_RBMC = 1 / diag_QTilde + jnp.mean(
            (
                1
                / diag_QTilde
                * (
                    v_QTilde(x_post_samples_demeaned)
                    - diag_QTilde * x_post_samples_demeaned
                )
            )
            ** 2,
            axis=0,
        )
        return var_x_post_samples_RBMC
