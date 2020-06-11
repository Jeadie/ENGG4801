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
        dataset = tf.data.TFRecordDataset(self.file_names).flat_map(map_func=self._parse_example)
            # , num_parallel_calls=multiprocessing.cpu_count())

        # Remove bad examples and enforce shape
        dataset = dataset.filter( lambda x, y: tf.math.reduce_min(x) != -10000)

        # Preprocess
        dataset = dataset.map(self.preprocess)
        # dataset = dataset.batch(batch_size=self.batch_size)
        return dataset

    def preprocess(self, image, label):
        """

        :param image:
        :param label:
        :return:
        """
        image, label = tf.reshape(image, [-1, 256, 256, 14]), tf.reshape(label, [-1, 3])
        # image = util.tf_equalize_histogram(image)
        # image = tf.clip_by_value(image, clip_value_min=-1, clip_value_max=100)
        # image, norms = tf.linalg.normalize(image)

        return image, label

    def _parse_example(self,_example: tf.Tensor) -> tf.data.Dataset:
        """ Parses a single saved TFRecord example and parses it to a set of elements to be added to the dataset.

        Flow:
          * Patient Metadata: parses patient label data from example.
          * Parses Images: parses all possible imaging tags that could be in the example.
          * Calculate Label: calculates the label to use, given the patients outcome data.
          * Create Dataset: creates a dataset object of elements from the Example.

        Args:
            example: the tfrecord for to read the data from

        Returns:
            a parsed input example and its respective label
        """
        # do parsing on the cpu
        with tf.device("/cpu:0"):
            results = {}

            # Patient metadata
            feature_groups = [
                 {"pCR": tf.io.FixedLenFeature((1), tf.int64, default_value=-1),
                  "RCB": tf.io.FixedLenFeature((1), tf.int64, default_value=-1)},
             ] + [
                dict([
                    (key, tf.io.VarLenFeature(tf.int64)),
                    (key.replace("image", "shape"), tf.io.FixedLenFeature([3], tf.int64, default_value=3*[-1]))])
                for key in util.possible_imaging_series_tags()]

            # Parses Images
            for features in feature_groups:
                try:
                    result = tf.io.parse_single_example(_example, features)
                    results.update(result)
                except Exception as e:
                    # Expected Behaviour
                    continue

            # Convert shape of images and filter images to get best.
            image_keys = list(filter(lambda x: "image" in x, results.keys()))
            images = []
            for image_key in image_keys:
                image = tf.py_function(
                        util.reconstruct_image, (results[image_key].values, results[image_key.replace("image", "shape")]),
                        tf.float32)
                images.append(image)

            images = tf.py_function(
                util.filter_images, (images),
                tf.float32)

            # Calculates Label
            group = tf.cast(util.calculate_group_from_results(results["pCR"], results["RCB"]), dtype=tf.uint8)
            group = tf.one_hot(group - 1, 3)[0, ...]

            # Create Dataset
            ds = tf.data.Dataset.from_tensor_slices(images)
            return ds.map(lambda x: (x, group))

            # return image, group

    def _augment(self, example: tf.Tensor) -> tf.Tensor:
        """
        Randomly augment the input image to try improve training variance
        Args:
            example: parsed input example
        Returns:
            The same input example but possibly augmented
        """
        # Convert Images
        # images = tf.linalg.normalize(images, axis=0)

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
