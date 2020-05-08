import os
import tempfile
import time
from typing import Dict, List, Tuple
import logging

from google.cloud import storage
import tensorflow as tf

from util import (
    calculate_group_from_results,
    get_images,
    bytes_feature,
    int64List_feature,
    required_imaging_series
)

_logger = logging.getLogger("Simplify Data")

DEFAULT_OUTPUT_SHARD_SIZE = 8

#TODO: get_required_image_keys
#TODO: Crop image into VOI.

def simplify_dataset(shards: List[str], output_prefix: str, shard_size: int=DEFAULT_OUTPUT_SHARD_SIZE ) -> None:
    """ Converts complete TFRecord shards and parses out the data needed for the
    pre-treatment imaging model to be trained.

    Gets all the series used in this model from the original, global shards, and
    saves them to the appropriate prefix. If the number of shards is not
    specified, the same number of shards are used as inputted.

    Args:
        shards: The list of TFrecord shards with the global data.
        output_prefix: The prefix to save the shards to. Saved as `output_prefix-0000X-of-00000{num_shards}.tfrecords`
        shard_size: The number of valid elements to save in each shard.

    Returns:
        None
    """
    shard_i = 0
    i = 0
    new_writer = True
    for raw in iter(tf.data.TFRecordDataset(shards)):
        if new_writer:
            shard_i += 1
            i = 0
            writer_file = f"{output_prefix}_{shard_i}.tfrecords"
            writer = tf.io.TFRecordWriter(writer_file)
            new_writer = False

        tensor_element = parse_raw(raw)

        # If element is suited for dataset, write to TFRecord.
        if tensor_element is not None:
            _logger.info("Writing to disk")
            writer.write(tensor_element.SerializeToString())
            i += 1
        else:
            _logger.warning("bad example")

        if i >= shard_size:
            writer.close()

            if not upload_to_gcs(writer_file):
                time.sleep(2)
                if not upload_to_gcs(writer_file):
                    _logger.error(f"Tried twice to upload file {writer_file}")

            new_writer = True

def simplify_dataset_gcs(gcs_prefix: str, shards: List[str], output_prefix: str, shard_size: int=DEFAULT_OUTPUT_SHARD_SIZE ) -> None:
    """ Converts complete TFRecord shards and parses out the data needed for the
    pre-treatment imaging model to be trained.

    Gets all the series used in this model from the original, global shards, and
    saves them to the appropriate prefix. If the number of shards is not
    specified, the same number of shards are used as inputted.

    Args:
        shards: The list of TFrecord shards with the global data.
        output_prefix: The prefix to save the shards to. Saved as `output_prefix-0000X-of-00000{num_shards}.tfrecords`
        shard_size: The number of valid elements to save in each shard.

    Returns:
        None
    """
    shard_i = 0
    i = 0
    new_writer = True
    writer = None
    for s in shards:
        with tempfile.TemporaryDirectory() as local_tmp:
            local_file = f"{str(local_tmp)}/{s}"
            download_blob(f"{gcs_prefix}{s}", local_file)
            dataset = tf.data.TFRecordDataset([local_file])

            for raw in dataset:
                if new_writer:
                    shard_i += 1
                    i = 0
                    writer_file = f"result_{shard_i}.tfrecords"
                    writer = tf.io.TFRecordWriter(writer_file)
                    new_writer = False

                tensor_element = parse_raw(raw)

                # If element is suited for dataset, write to TFRecord.
                if tensor_element is not None:
                    _logger.info("Writing to disk")
                    writer.write(tensor_element.SerializeToString())
                    writer.flush()
                    i += 1

                del tensor_element
                if i >= shard_size:
                    writer.flush()
                    writer.close()

                    if not upload_to_gcs(writer_file, prefix=output_prefix):
                        time.sleep(2)
                        if not upload_to_gcs(writer_file):
                            _logger.error(f"Tried twice to upload file {writer_file}")

                    new_writer = True
                    del writer
            del dataset

    if not new_writer:
        if not upload_to_gcs(writer_file, prefix=output_prefix):
            time.sleep(2)
            if not upload_to_gcs(writer_file):
                _logger.error(f"Tried twice to upload file {writer_file}")

def download_blob(filename: str,  save_path: str, bucket_name:str="ispy_dataquery") -> bool:
    """
    """
    try:
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(filename)
        blob.download_to_filename(save_path)

    except Exception as e:
        _logger.warning(f"Failed to download: gs://{bucket_name}/{filename} to {save_path}. Error: {str(e)}.")
        return False
    else:
        _logger.info(f"Successfullt downloaded result TFRecord: gs://{bucket_name}/{filename} to {save_path}.")
        return True

def upload_to_gcs(filename: str, bucket_name:str="ispy_dataquery", prefix: str="") -> bool:
    try:
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(f"{prefix}{filename}")
        blob.upload_from_filename(filename=filename)
        os.remove(filename)

    except Exception as e:
        _logger.warning(f"Failed to upload result TFRecord: {filename} to gs://{bucket_name}/{prefix}{filename}. Error: {str(e)}.")
        return False
    else:
        _logger.info(f"Succeeded to upload result TFRecord: {filename} to gs://{bucket_name}/{prefix}{filename}.")
        return True


def parse_raw(input) -> tf.Tensor:
    """ Converts a raw Patient proto to a Tensor.

    Args:
        input:

    Returns
        A Tensor
    """
    patient, images = get_images(input)
    images = dict(images)

    # Check if example has necessary keys. If not, return None.
    image_keys = get_required_image_keys(list(images.keys()))
    if not image_keys:
        return None

    patient["group"] = calculate_group_from_results(patient["pCR"], patient["RCB"])
    for k in patient.keys():
        patient[k] = int64List_feature([patient[k]])

    for k in image_keys:
        images[k.replace("image", "shape")] = int64List_feature(images[k].shape)
        images[k] = int64List_feature(images[k].numpy().flatten().tolist())

    patient.update(images)

    return tf.train.Example(features=tf.train.Features(feature=patient))


def get_required_image_keys(keys: List[str]) -> List[str]:
    """Gets the required keys for an example given a list of imaging names. If
    the minimum keys are not present, an empty list is returned.

    Args:
        keys:

    Returns:
    """
    if set(required_imaging_series()) <= set(keys):
        return list(set(keys).intersection(set(required_imaging_series())))
    else:
        return []


if __name__ == '__main__':
    simplify_dataset_gcs("output/", [f"small_result_2-00000-of-00001.tfrecords_{str(i)}" for i in range(41)], os.environ.get("GCS_UPLOAD_PREFIX", "simplify/"))

