import jax
from jax import numpy as np
from jax.nn import sigmoid

def normal_sim_binary(rng, n, d, k, beta=None, gamma=None):

    rng, next_rng = jax.random.split(rng)
    X = jax.random.normal(next_rng, shape=(n, d))

    if beta is None:
        rng, next_rng = jax.random.split(rng)
        beta = jax.random.normal(next_rng, shape=(d ,1))

    if gamma is None:
        rng, next_rng = jax.random.split(rng)
        gamma = jax.random.normal(next_rng, shape=(d, k))

    rng, next_rng = jax.random.split(rng)
    g = jax.random.categorical(rng, X @ gamma)

    p = sigmoid(X @ beta)
    rng, next_rng = jax.random.split(rng)
    Y = jax.random.bernoulli(next_rng, p)

    group_indices = {int(group): np.where(group == g)[0] for group in range(k)}
    group_Xs = jax.tree_util.tree_map(lambda idx: X[idx], group_indices)
    group_Ys = jax.tree_util.tree_map(lambda idx: np.sum(Y[idx]), group_indices)
    group_Ns = jax.tree_util.tree_map(lambda idx: idx.shape[0], group_indices)
    return beta, gamma, X, Y, (group_Xs, group_Ys, group_Ns)


def normal_sim_categorical(rng, n, d, k, p, eps=1e-6, beta=None, gamma=None):

    rng, next_rng = jax.random.split(rng)
    X = jax.random.normal(next_rng, shape=(n, d - 1))
    X = np.hstack([np.ones((n, 1)), X])

    if beta is None:
        rng, next_rng = jax.random.split(rng)
        beta = jax.random.normal(next_rng, shape=(d, p - 1))
        beta = np.concatenate([beta, np.zeros((d, 1))], axis=-1)

    if gamma is None:
        rng, next_rng = jax.random.split(rng)
        gamma = jax.random.normal(next_rng, shape=(d, k))

    rng, next_rng = jax.random.split(rng)
    g = jax.random.categorical(rng, X @ gamma)

    rng, next_rng = jax.random.split(rng)
    Y = jax.random.categorical(rng, np.tensordot(X, beta, axes=1))
    Y = jax.nn.one_hot(Y, p)

    group_indices = {int(group): np.where(group == g)[0] for group in range(k)}
    group_Xs = jax.tree_util.tree_map(lambda idx: X[idx], group_indices)
    group_Ys = jax.tree_util.tree_map(lambda idx: np.sum(Y[idx], axis=0), group_indices)
    group_Ns = jax.tree_util.tree_map(lambda idx: idx.shape[0], group_indices)
    return beta, gamma, X, Y, (group_Xs, group_Ys, group_Ns)