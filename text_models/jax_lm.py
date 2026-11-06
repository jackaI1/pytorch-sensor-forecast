"""A small character language model with synchronous data-parallel updates."""

import argparse
import pickle
from pathlib import Path
from urllib.request import urlopen

import jax
import jax.numpy as jnp
import numpy as np

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def init(key, vocab, width=64, length=64):
    keys = iter(jax.random.split(key, 9))

    def weight(shape):
        return jax.random.normal(next(keys), shape) * 0.02

    return dict(
        embed=weight((vocab, width)),
        pos=weight((length, width)),
        q=weight((width, width)),
        k=weight((width, width)),
        v=weight((width, width)),
        o=weight((width, width)),
        ff1=weight((width, width * 4)),
        ff2=weight((width * 4, width)),
        head=weight((width, vocab)),
    )


def norm(x):
    return (x - x.mean(-1, keepdims=True)) / jnp.sqrt(x.var(-1, keepdims=True) + 1e-5)


def logits(p, x):
    h = p["embed"][x] + p["pos"][: x.shape[1]]
    z = norm(h)
    q, k, v = z @ p["q"], z @ p["k"], z @ p["v"]
    scores = q @ jnp.swapaxes(k, -1, -2) / jnp.sqrt(q.shape[-1])
    mask = jnp.tril(jnp.ones((x.shape[1], x.shape[1]), dtype=bool))
    h = h + jax.nn.softmax(jnp.where(mask, scores, -1e9), axis=-1) @ v @ p["o"]
    h = h + jax.nn.gelu(norm(h) @ p["ff1"]) @ p["ff2"]
    return norm(h) @ p["head"]


def loss(p, x, y):
    log_probs = jax.nn.log_softmax(logits(p, x), axis=-1)
    return -jnp.take_along_axis(log_probs, y[..., None], axis=-1).mean()


def _shard(params, inputs, targets):
    value, grad = jax.value_and_grad(loss)(params, inputs, targets)
    return jax.lax.pmean(value, "devices"), jax.lax.pmean(grad, "devices")


parallel_grad = jax.pmap(_shard, axis_name="devices", in_axes=(None, 0, 0))


def batch(data, rng, count, length):
    starts = rng.integers(0, len(data) - length, size=count)
    tokens = np.stack([data[i : i + length + 1] for i in starts])
    return tokens[:, :-1], tokens[:, 1:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/tinyshakespeare.txt"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/jax-lm.pkl"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.data.exists():
        args.data.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(URL, timeout=30) as response:
            args.data.write_bytes(response.read())
    text = args.data.read_text(encoding="utf-8")
    chars = sorted(set(text))
    ids = np.array([chars.index(c) for c in text], dtype=np.int32)
    split = int(len(ids) * 0.9)
    train, valid = ids[:split], ids[split:]
    rng = np.random.default_rng(42)
    p = init(jax.random.PRNGKey(42), len(chars))
    step = 0
    if args.resume:
        with args.checkpoint.open("rb") as f:
            state = pickle.load(f)
        if state["chars"] != chars:
            raise ValueError("Checkpoint vocabulary differs from the corpus")
        p, step = state["params"], state["step"]
        rng.bit_generator.state = state["rng"]
    devices = jax.local_device_count()
    if args.batch_size % devices:
        raise ValueError("batch size must be divisible by device count")
    for _ in range(args.steps):
        x, y = batch(train, rng, args.batch_size, 64)
        x = x.reshape(devices, -1, 64)
        y = y.reshape(devices, -1, 64)
        value, grads = parallel_grad(p, x, y)
        p = jax.tree.map(lambda a, g: a - 0.1 * g[0], p, grads)
        step += 1
    x, y = batch(valid, np.random.default_rng(123), 128, 64)
    val_loss = float(loss(p, x, y))
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with args.checkpoint.open("wb") as f:
        pickle.dump(
            dict(params=jax.device_get(p), step=step, chars=chars, rng=rng.bit_generator.state), f
        )
    print(
        f"step={step} devices={devices} validation_loss={val_loss:.4f} perplexity={np.exp(val_loss):.2f}"
    )


if __name__ == "__main__":
    main()
