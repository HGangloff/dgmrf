"""
Main training loop
"""

from typing import Callable
import equinox as eqx
import jax
import jax.numpy as jnp
from optax import GradientTransformation, OptState
from jaxtyping import Array, Int, Float, Key

from dgmrf.models._dgmrf import DGMRF
from dgmrf.models._variational_distributions import VariationalDistribution


class Model(eqx.Module):
    dgmrf: DGMRF
    posterior: VariationalDistribution
    noise_parameter: Array

    # TODO wrap the DGMRF methods in the Model class to offer less verbose
    # calls
    # def posterior_samples():
    #    self.dgmrf.posterior_samples(...)
    # def get_post_mu():
    # ...


def train_loop(
    loss_fn: Callable,
    model: Model,
    y: Array,
    n_iter: Int,
    tx: GradientTransformation,
    opt_state: OptState,
    key: Key,
    print_rate: Int,
    *args,
    **kwargs,
) -> tuple[Model, Array, OptState]:
    """
    Main training loop
    """

    @eqx.filter_jit
    def make_step(
        model: Model, key: Key, opt_state: OptState
    ) -> tuple[Model, Key, OptState, Float]:
        key, subkey = jax.random.split(key, 2)
        loss_value, grads = eqx.filter_value_and_grad(loss_fn)(
            model, y, subkey, *args, **kwargs
        )
        updates, opt_state = tx.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, key, opt_state, loss_value

    loss_values = []
    for i in range(n_iter):
        model, key, opt_state, loss_value = make_step(model, key, opt_state)

        loss_values.append(-loss_value)
        if i % print_rate == 0:
            print(f"Iteration {i}, loss_value = {-loss_value}")

    print(f"End of training (iteration {n_iter}), loss_value = {-loss_value}")

    return model, jnp.array(loss_values), OptState
