from typing import Dict, List, Tuple
import logging

import tensorflow as tf

import data_loader.util import get_images, int64List_feature

_logger = logging.getLogger("Simplify Data")

DEFAULT_OUTPUT_SHARD_SIZE = 20

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
            writer = tf.io.TFRecordWriter(f"{output_prefix}_{shard_i}.tfrecords")
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
            new_writer = True


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

    for k in images.keys():
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
    return keys

def calculate_group_from_results(pCR: int, RCB: int) -> int:
    """ Calculates the response group of the patient from their pCR and RCB result.

    Args:
        pCR: Boolean integer of whether the patient had a complete response to NAC.
        RCB: The residual cancer burden of the patient. [0, 4]

    Returns: 
        The integer, g representing the classification of the patient. 1 <= g <=3.
    """
    if pCR:
        return 1
    if RCB <= 2:
        return 2
    return 3

if __name__ == '__main__':
    simplify_dataset(["1020_result-00000-of-00001.tfrecords"], "test-out")