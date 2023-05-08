import jax
from jax import numpy as np
from jax.nn import sigmoid

def normal_sim(rng, n, d, k, eps=1e-6):

    rng, next_rng = jax.random.split(rng)
    X = jax.random.normal(next_rng, shape=(n, d))
    rng, next_rng = jax.random.split(rng)
    beta = jax.random.normal(next_rng, shape=(d ,1))
    rng, next_rng = jax.random.split(rng)
    gamma = jax.random.normal(next_rng, shape=(d, k))

    rng, next_rng = jax.random.split(rng)
    g = jax.random.categorical(rng, X @ gamma)

    p = sigmoid(X @ beta)
    p = (1 - eps) * p + eps / 2
    rng, next_rng = jax.random.split(rng)
    Y = jax.random.bernoulli(next_rng, p)

    group_indices = {int(group): np.where(group == g)[0] for group in range(k)}
    group_Xs = jax.tree_util.tree_map(lambda idx: X[idx], group_indices)
    group_Ys = jax.tree_util.tree_map(lambda idx: np.sum(Y[idx]), group_indices)
    group_Ns = jax.tree_util.tree_map(lambda idx: idx.shape[0], group_indices)
    return beta, X, Y, (group_Xs, group_Ys, group_Ns)

