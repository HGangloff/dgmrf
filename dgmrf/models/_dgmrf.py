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
            A tuple of jnp.array to designate the (Adjacency, Degree) matrices to create a graph DGMRF. Cannot
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
                        )
                    )
                else:
                    self.key, subkey = jax.random.split(self.key, 2)
                    self.layers.append(
                        ConvLayer(
                            jax.random.uniform(subkey, (7,), minval=-1, maxval=1),
                            height_width[0],
                            height_width[1],
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
                            jax.random.uniform(subkey1, (4,), minval=-1, maxval=1),
                            A_D[0],
                            A_D[1],
                            *args,
                            **kwargs,
                            key=subkey2,
                        )
                    )
                self.N = A_D[0].shape[0]

    def __call__(self, x, transpose=False):
        """
        Return the composition of g = gLgL-1...g1(z0) with z0 = x
        """
        z_l_1 = x
        for l in range(self.nb_layers):
            z_l_1 = self.layers[l](z_l_1, transpose=transpose)
        return z_l_1

    def log_det(self):
        """
        Compute the log determinant
        """
        log_det = 0
        for l in range(self.nb_layers):
            log_det += self.layers[l].efficient_logdet_G_l()
        return log_det

    def get_Q(self):
        """
        Get the precision matrix of the DGMRF using the formula Q = G^TG
        """
        G = self.layers[0].get_G()
        for _ in range(self.nb_layers - 1):
            G = G @ self.layers[0].get_G()
        return G.T @ G

    def get_mu(self):
        """
        mu = G^-1*b = G^-1*g(0). We invert G with conjugate gradient
        """

        def G(x):
            return self(x)

        mu, _ = jax.scipy.sparse.linalg.cg(
            G, self(jnp.zeros((self.N,))), x0=jnp.zeros((self.N,))
        )
        return mu

    def sample(self, key):
        """
        Sample from the DGMRF
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
