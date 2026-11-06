"""Train and export an IMDB review classifier."""

import argparse
from pathlib import Path

import tensorflow as tf

VOCAB = 10000
LENGTH = 128


def build_model():
    inputs = tf.keras.Input(shape=(LENGTH,), dtype=tf.int32, name="tokens")
    positions = tf.range(LENGTH)
    x = tf.keras.layers.Embedding(VOCAB, 32)(inputs) + tf.keras.layers.Embedding(LENGTH, 32)(
        positions
    )
    attention = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=16)(x, x)
    x = tf.keras.layers.LayerNormalization()(x + attention)
    ff = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.LayerNormalization()(x + tf.keras.layers.Dense(32)(ff))
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="probability")(x)
    model = tf.keras.Model(inputs, output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def data(limit):
    (x, y), (xt, yt) = tf.keras.datasets.imdb.load_data(num_words=VOCAB)

    def pad(a):
        return tf.keras.utils.pad_sequences(
            a, maxlen=LENGTH, padding="post", truncating="post"
        ).astype("int32")

    return pad(x[:limit]), y[:limit], pad(xt[: min(limit, len(xt))]), yt[: min(limit, len(yt))]


def export(model, path):
    class Serving(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec([None, LENGTH], tf.int32, name="tokens")])
        def serve(self, tokens):
            return {"probability": model(tokens, training=False)}

    module = Serving()
    module.model = model
    tf.saved_model.save(module, str(path), signatures={"serving_default": module.serve})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--export", type=Path, default=Path("outputs/review_model/1"))
    args = parser.parse_args()
    tf.keras.utils.set_random_seed(42)
    x, y, xt, yt = data(args.limit)
    model = build_model()
    model.fit(x, y, batch_size=64, epochs=args.epochs, verbose=2)
    metrics = model.evaluate(xt, yt, verbose=0)
    export(model, args.export)
    print(f"test_loss={metrics[0]:.4f} test_accuracy={metrics[1]:.4f} n={len(yt)}")


if __name__ == "__main__":
    main()
