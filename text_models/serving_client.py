"""Send token batches to TensorFlow Serving and time the requests."""

import argparse
import json
import time
from urllib.request import Request, urlopen

import numpy as np


def predict(url, tokens):
    request = Request(
        url,
        data=json.dumps(
            {"signature_name": "serving_default", "instances": np.asarray(tokens).tolist()}
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=30) as response:
        return np.asarray(json.load(response)["predictions"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8501/v1/models/review:predict")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--requests", type=int, default=20)
    args = parser.parse_args()
    tokens = np.zeros((args.batch, 128), dtype=np.int32)
    predict(args.url, tokens)
    times = []
    for _ in range(args.requests):
        start = time.perf_counter()
        predict(args.url, tokens)
        times.append((time.perf_counter() - start) * 1000)
    print(
        f"batch={args.batch} requests={args.requests} mean_ms={np.mean(times):.2f} p95_ms={np.percentile(times, 95):.2f}"
    )


if __name__ == "__main__":
    main()
