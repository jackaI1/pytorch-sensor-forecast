"""Prepare a trained model or compare its saved predictions with the REST service."""

import argparse
import time
from pathlib import Path
from urllib.error import HTTPError

import numpy as np
import tensorflow as tf

from text_models import serving_client, tf_classifier

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["prepare", "check"])
args = parser.parse_args()
path = Path("outputs/serving_check.npz")
if args.mode == "prepare":
    tf.keras.utils.set_random_seed(42)
    x, y, xt, yt = tf_classifier.data(256)
    model = tf_classifier.build_model()
    model.fit(x, y, epochs=1, batch_size=32, verbose=2)
    loss, accuracy = model.evaluate(xt, yt, verbose=0)
    tf_classifier.export(model, Path("outputs/review_model/1"))
    np.savez(path, x=xt[:16], expected=model(xt[:16], training=False).numpy())
    print(f"IMDB held-out n={len(yt)} loss={loss:.4f} accuracy={accuracy:.4f}")
else:
    data = np.load(path)
    url = "http://localhost:8501/v1/models/review:predict"
    for attempt in range(60):
        try:
            actual = serving_client.predict(url, data["x"])
            break
        except HTTPError as exc:
            if exc.code != 503:
                raise RuntimeError(exc.read().decode()) from exc
            time.sleep(1)
        except OSError:
            time.sleep(1)
    else:
        raise RuntimeError("TF Serving did not become ready")
    np.testing.assert_allclose(actual, data["expected"], rtol=1e-5, atol=1e-6)
    times = []
    for _ in range(20):
        start = time.perf_counter()
        serving_client.predict(url, data["x"])
        times.append((time.perf_counter() - start) * 1000)
    print(
        f"REST matches in-process: batch=16, max_abs_error={np.max(np.abs(actual - data['expected'])):.8f}, mean_ms={np.mean(times):.2f}, p95_ms={np.percentile(times, 95):.2f}"
    )
