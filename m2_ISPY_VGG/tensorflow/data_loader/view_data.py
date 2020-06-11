import sys
from typing import Dict, List, Tuple
import logging
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

from data_loader.util import get_images

_logger = logging.getLogger("View Data")


def main(args: List[str]) -> None:
    """ Attempts to parse all the Images from a list TFRecords and displays
    each Series as an animated image of the 3D stack.

    Args:
        args: `view_data.py` followed by a list of TFRecord paths.
            e.g. ["view_data.py", "data_1.tfrecords", "data_2.tfrecords"]

    Return:
        None
    """
    for raw in iter(tf.data.TFRecordDataset(args[1:])):
        patient_details, image_stacks = get_images(raw)
        _logger.info(f"Patient details: {patient_details}")
        for name, image in image_stacks:
            visualise_image(name, image)


def visualise_image(name: str, image: np.array) -> None:
    """ Visualises a 3D stack of images in grayscale.

    Runs through each slice of the 3D image with a slight pause between to
    create an animation of the 3D stack.

    Args:
        name: The name of the 3D image to be displayed above the animation.
        image: A 3D image whereby the first index is the z dimension.
            i.e [i, :,:] is a 2D slice of the 3D image.

    Returns:
        None
    """
    image = tf.cast(image, tf.int32)
    for j in range(image.shape[0]):
        plt.imshow(image[j, ...], cmap="gray")
        plt.title(f"Name of series: {name}")
        plt.pause(0.001)
    plt.show()




if __name__ == "__main__":
    main(sys.argv)
