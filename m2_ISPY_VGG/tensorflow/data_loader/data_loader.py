import multiprocessing
import os
import random
from typing import Dict, List, Tuple

import tensorflow as tf
import absl

from base.data_loader import DataLoader
import data_loader.util as util

_logger = absl.logging

class TFRecordShardLoader(DataLoader):
    SHARD_SUFFIX = ".tfrecords"

    def __init__(self, config: Dict, mode: str) -> None:
        """
        An example of how to create a dataset using tfrecords inputs
        :param config: global configuration
        :param mode: current training mode (train, test, predict)
        """
        super().__init__(config, mode)

        # Get a list of files in case you are using multiple tfrecords
        if self.mode == "train":
            self.file_names = TFRecordShardLoader.get_file_or_files(self.config["train_files"])
            self.batch_size = self.config["train_batch_size"]
        elif self.mode == "val":
            self.file_names = TFRecordShardLoader.get_file_or_files(self.config["eval_files"])
            self.batch_size = self.config["eval_batch_size"]
        else:
            self.file_names = TFRecordShardLoader.get_file_or_files(self.config["test_files"])
            self.batch_size = 1

    @staticmethod
    def get_file_or_files(paths: List[str]) -> List[str]:
        """ Gets all files in a folder or the single file itself.

        Args:
            path: A list of paths to include in the dataset. Each path is
              either the path to a file or directory.

        Returns:
            If a file and valid suffix, returns the file in a list. Otherwise
            returns all files in the directory with appropriate suffix.
        """
        files = []
        for p in paths:
            if os.path.isdir(p):
                files.extend([f"{p}{x}" for x in os.listdir(p)])
            else:
                files.append(p)
        return list(filter(lambda x: x.endswith(TFRecordShardLoader.SHARD_SUFFIX), files))

    def input_fn(self) -> tf.data.Dataset:
        """
        Create a tf.Dataset using tfrecords as inputs, use parallel loading
        and augmentation using the CPU to reduce bottle necking of operations
        on the GPU.

        Returns:
             A Dataset ready for use in the model.
        """
        dataset = tf.data.TFRecordDataset(self.file_names).map(
            map_func=self._parse_example, num_parallel_calls=multiprocessing.cpu_count()
        )
        # Remove bad examples and enforce shape
        dataset = dataset.filter( lambda x, y: tf.math.reduce_min(x) != -10000)

        # Not batching yet, expand dims.
        dataset = dataset.map(lambda image, label: (tf.reshape(image, [-1, 256, 256, 1]), tf.reshape(label, [-1, 3])))
        # dataset = dataset.batch(batch_size=self.batch_size)
        return dataset

    def _parse_example(
            self, _example: tf.Tensor
    ) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
        """
        Used to read in a single example fr om a tf record file and do any augmentations necessary
        :param example: the tfrecord for to read the data from
        :return: a parsed input example and its respective label
        """
        # tf.data.Dataset.from_tensor_slices(
        # do parsing on the cpu
        with tf.device("/cpu:0"):
            results = {}

            # Break up required features into groups that may exist together.
            feature_groups = [
                 {"pCR": tf.io.FixedLenFeature((1), tf.int64, default_value=-1),
                  "RCB": tf.io.FixedLenFeature((1), tf.int64, default_value=-1)},
             ] + [
                dict([
                    (key, tf.io.VarLenFeature(tf.int64)),
                    (key.replace("image", "shape"), tf.io.FixedLenFeature([3], tf.int64, default_value=3*[-1]))])
                for key in util.possible_imaging_series_tags()]

            # Parse group by group to avoid a single example not having all specific keys
            for features in feature_groups:
                try:
                    result = tf.io.parse_single_example(_example, features)
                    results.update(result)
                except Exception as e:
                    # Expected Behaviour
                    continue

            # Get label convert to one-hot
            group = tf.cast(util.calculate_group_from_results(results["pCR"], results["RCB"]), dtype=tf.uint8)
            group = tf.one_hot(group - 1, 3)[0, ...]

            # Convert shape of images and filter images to get best.
            image_keys = list(filter(lambda x: "image" in x, results.keys()))
            images = []
            for image_key in image_keys:
                image = tf.py_function(
                        util.reconstruct_image, (results[image_key].values, results[image_key.replace("image", "shape")]),
                        tf.float32)
                images.append(image)

            image = tf.py_function(
                util.filter_images, (images),
                tf.float32)

            return image, group

    @staticmethod
    def _normalise(example: tf.Tensor) -> tf.Tensor:
        """ Normalise a 3D image as

        Args:
            example:
        :return:
        """
        return example

    @staticmethod
    def _augment(example: tf.Tensor) -> tf.Tensor:
        """
        Randomly augment the input image to try improve training variance
        Args:
            example: parsed input example
        Returns:
            The same input example but possibly augmented
        """
        # random rotation
        if random.uniform(0, 1) > 0.5:
            example = tf.contrib.image.rotate(
                example, tf.random_uniform((), minval=-0.2, maxval=0.2)
            )
        # random noise
        if random.uniform(0, 1) > 0.5:
            # assumes values are normalised between 0 and 1
            noise = tf.random_normal(
                shape=tf.shape(example), mean=0.0, stddev=0.2, dtype=tf.float32
            )
            example = example + noise
            example = tf.clip_by_value(example, 0.0, 1.0)
            # random flip
            example = tf.image.random_flip_up_down(example)
        return tf.image.random_flip_left_right(example)

    def __len__(self) -> int:
        """
        Get number of records in the dataset
        :return: number of samples in all tfrecord files
        """
        return sum(
            1 for fn in self.file_names for _ in tf.compat.v1.python_io.tf_record_iterator(fn)
        )
