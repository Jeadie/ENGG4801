import multiprocessing
import os
import random
from typing import Dict, List, Tuple

import tensorflow as tf

from base.data_loader import DataLoader
import data_loader.util as util


class TFRecordShardLoader(DataLoader):
    SHARD_SUFFIX = ".tfrecords"
    def __init__(self, config: dict, mode: str) -> None:
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
        for p in paths:
            if os.path.isdir(p):
                files = os.listdir(p)
            else:
                files = [p]
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

        # only shuffle training data
        if self.mode == "train":
            # shuffles and repeats a Dataset returning a new permutation for each epoch. with serialised compatibility
            dataset = dataset.shuffle(max(len(self) // self.config["train_batch_size"], 2))
        else:
            dataset = dataset.repeat(self.config["num_epochs"])
        # create batches of data
        print(max(len(self) // self.config["train_batch_size"], 2), "self.batch_size", self.batch_size)
        dataset = dataset.batch(batch_size=self.batch_size)
        return dataset

    def _parse_example(
            self, _example: tf.Tensor
    ) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
        """
        Used to read in a single example from a tf record file and do any augmentations necessary
        :param example: the tfrecord for to read the data from
        :return: a parsed input example and its respective label
        """
        # do parsing on the cpu
        with tf.device("/cpu:0"):
            features = {
                **{"group": tf.io.FixedLenFeature(shape=[1], dtype=tf.int64)},
                **dict([
                    (key, tf.io.VarLenFeature(tf.int64)) for key in util.required_imaging_series()
                ]),
                **dict([
                    (key.replace("image", "shape"), tf.io.FixedLenFeature([3], tf.int64)) for key in
                    util.required_imaging_series()
                ]),
            }
            example = tf.io.parse_single_example(_example, features)

            result = {}
            for image_key in util.required_imaging_series():
                result[image_key] = tf.py_function(
                    util.reconstruct_image, (example[image_key].values, example[image_key.replace("image", "shape")]),
                    tf.int64)
            print("result", result)
            image = result["time1/MRI_DCE/image"]

            # TODO: This must magically pick a single slice from 3D (eventually will have to pick 64x64 too.
            image = tf.transpose(image, [2, 1, 0])
            image = tf.expand_dims(image[..., 0], -1)
            return image, tf.one_hot(example["group"] - 1, 3)[0, ...]


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
