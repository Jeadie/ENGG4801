from typing import Tuple

import tensorflow as tf
import absl

from data_loader.data_loader import TFRecordShardLoader
from data_loader.util import floatList_feature, int64List_feature, tf_equalize_histogram

_logger = absl.logging


class ResampledTFRecordDataset(TFRecordShardLoader):

    def input_fn(self) -> tf.data.Dataset:
        """
        Create a tf.Dataset using tfrecords as inputs, use parallel loading
        and augmentation using the CPU to reduce bottle necking of operations
        on the GPU.

        Returns:
             A Dataset ready for use in the model.
        """
        dataset = tf.data.TFRecordDataset(self.file_names).map(map_func=self.deserialise_example)

        # Preprocess
        dataset = dataset.map(self.preprocess)
        dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=False)
        return dataset

    def preprocess(self, image, label):
        if self.config.get("histogram_equalise", True):
            image = tf.py_function(
                tf_equalize_histogram,
                (image, ),
                tf.float32)
        return tf.reshape(image, [256, 256, 50]), label

    def deserialise_example(self, proto, shape=(256, 256, 50)) -> tf.Tensor:
        feature_description = {
            'image': tf.io.FixedLenFeature(shape, tf.float32),
            'label': tf.io.FixedLenFeature([3], tf.int64),

        }

        example = tf.io.parse_single_example(proto, feature_description)
        return (example["image"], example["label"])


def serialise_example(image, label) -> str:
    """

    Function used in eagerly execute (e.g. in a pyfunction)

    Args:
        image:
        label:

    Returns:
         A tf.train.Example serialised to a string.
    """
    print(image.shape)
    feature = {
        'image': floatList_feature(image.numpy().flatten().tolist()),
        'label': int64List_feature(tf.cast(label, tf.uint8).numpy().flatten().tolist()),
    }
    proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return proto.SerializeToString()

def tf_serialise_example(image, label) -> str:
    """

    Args:
        image:
        label:

    Returns:
         An example serialised to string.
    """
    tf_string = tf.py_function(
        serialise_example,
        (image, label),
        tf.string)
    return tf.reshape(tf_string, ())

def main():
    config = {
        "test_files": ["output/"],
        "use_stack": False,
        "output_filename": "output/combined_total.tfrecord",
        "histogram_equalise": False, # Don't remove information when saving as source data. (fast-ish operation)
        "interpolate": True,
        "num_slices": 50,
    }

    print(f"Beginning to parse dataset.")
    _logger.info(f"Beginning to parse dataset.")
    ds = TFRecordShardLoader(config, mode="test")
    ds = ds.input_fn()

    _logger.info(f"Serialise dataset")
    print(f"Serialise dataset")
    serialised_ds = ds.map(tf_serialise_example)

    _logger.info(f"Saving TFRecord to {config.get('output_filename')}")
    print(f"Saving TFRecord to {config.get('output_filename')}")
    writer = tf.data.experimental.TFRecordWriter(config.get("output_filename"))
    writer.write(serialised_ds)
    _logger.info("Done")
    print("Done")


if __name__ == '__main__':
    main()
