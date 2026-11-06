import jax
import numpy as np

from text_models import jax_lm as lm


def test_parallel_loss_and_gradients_match():
    assert jax.local_device_count() >= 2
    p = lm.init(jax.random.PRNGKey(1), 16, width=8, length=8)
    x = np.random.default_rng(3).integers(0, 16, (8, 8), dtype=np.int32)
    y = np.roll(x, -1, axis=1)
    single_loss, single_grad = jax.value_and_grad(lm.loss)(p, x, y)
    n = jax.local_device_count()
    distributed_loss, distributed_grad = lm.parallel_grad(
        p, x.reshape(n, -1, 8), y.reshape(n, -1, 8)
    )
    np.testing.assert_allclose(distributed_loss, single_loss, rtol=1e-5, atol=1e-6)
    for expected, actual in zip(jax.tree.leaves(single_grad), jax.tree.leaves(distributed_grad)):
        np.testing.assert_allclose(actual[0], expected, rtol=1e-4, atol=1e-6)


def test_causal_mask():
    p = lm.init(jax.random.PRNGKey(0), 10, width=8, length=4)
    a = np.array([[1, 2, 3, 4]])
    b = np.array([[1, 2, 8, 9]])
    np.testing.assert_allclose(lm.logits(p, a)[:, :2], lm.logits(p, b)[:, :2], atol=1e-6)
