import numpy as np
import tensorflow as tf

from text_models import tf_classifier as classifier


def test_training_and_export(tmp_path):
    tf.keras.utils.set_random_seed(42)
    model = classifier.build_model()
    x = np.random.default_rng(1).integers(
        0, classifier.VOCAB, (8, classifier.LENGTH), dtype=np.int32
    )
    y = np.arange(8) % 2
    model.train_on_batch(x, y)
    expected = model(x, training=False).numpy()
    classifier.export(model, tmp_path / "1")
    loaded = tf.saved_model.load(str(tmp_path / "1")).signatures["serving_default"]
    actual = loaded(tokens=tf.constant(x))["probability"].numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
