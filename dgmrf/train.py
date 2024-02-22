"""
Main training loop
"""

import jax
import jax.numpy as jnp
from jax_tqdm import scan_tqdm
import optax


def train_loop(
    loss_fn, y, n_iter, params_init, static, tx, opt_state, key, *args, **kwargs
):
    """
    Main training loop

    Parameters
    ----------
    TODO

    Returns
    -------
    carry[1]
        The final parameter values
    loss_values
        The loss values for each gradient step
    """

    @scan_tqdm(n_iter)
    def scan_fun(carry, _):
        key, params, opt_state = carry
        key, subkey = jax.random.split(key, 2)
        loss, grads = jax.value_and_grad(loss_fn)(
            params, static, y, subkey, *args, **kwargs
        )
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (key, params, opt_state), loss

    key, subkey = jax.random.split(key, 2)
    carry, loss_values = jax.lax.scan(
        scan_fun, (subkey, params_init, opt_state), jnp.arange(n_iter)
    )
    return carry[1], loss_values
