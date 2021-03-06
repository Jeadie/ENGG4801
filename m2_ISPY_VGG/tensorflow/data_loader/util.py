from typing import Dict, List, Tuple, Union

import absl
import numpy as np
import tensorflow as tf

_logger = absl.logging


def int64List_feature(value: List[int]) -> tf.train.Feature:
    """ Converts an integer List into a int64_list Tensorflow feature.
        """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value: Union[bytes, str]) -> tf.train.Feature:
    """ Converts a byte or string data type into a bytes_list Tensorflow feature.
    """
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def get_images(raw_example: tf.Tensor) -> Tuple[Dict[str, int], List[Tuple[str, np.array]]]:
    """ Parses a single Patient proto from a TFRecord and retrieves specific
    patient data and all the images from the proto.

    Args:
        raw_example: A Tensor of type tf.string whereby the Tensor's value is
            the string proto of a tf.Example. i.e. One can create a tf.Example
            from the Tensor's value.

    Returns: Two elements in a Tuple:
        1. A mapping of the patient specific metadata. Currently, these fields
            are: `patient_id`, `pCR`, `RCB`.
        2. A list of 3D nd-array images for the patient. Each element consists
            of (as a Tuple) the name of the imaging Series (as found in the
            `raw_example`) and a 3D np.array of the image with shape as
            defined by its shape (again specified in the `raw_example`).
            e.g. ("timeX/MRI_RIGHT/image", np.array([[0,0,0...],..]]) )
    """
    string_example = tf.train.Example.FromString(raw_example.numpy())
    keys = list(
        filter(lambda x: "/image" in x, dict(string_example.features.feature).keys())
    )
    _logger.error(f"KEYS: {keys}")
    try:
        patient_features = {
            "patient_id": tf.io.FixedLenFeature((1), tf.int64, default_value=-1),
            "pCR": tf.io.FixedLenFeature((1), tf.int64, default_value=-1),
            "RCB": tf.io.FixedLenFeature((1), tf.int64, default_value=-1),
        }
        patient_details = tf.io.parse_example(raw_example, patient_features)

        if -1 in [patient_details[value].numpy()[0] for value in ["patient_id", "pCR", "RCB"]]:
            raise ValueError(f"Did not have correct values: {patient_details}. ")

    except (ValueError, tf.errors.InvalidArgumentError) as e:
        _logger.error(
            f"Could not pass patient data. Error: {e}. "
        )
        return {}, []

    # Convert Features to ints.
    for k in patient_features:
        patient_details[k] = list(patient_details[k].numpy())[0]

    images = []
    # Parse_example image by image in case of error, can keep processing other images.
    for k in keys:
        try:
            features = {
                k: tf.io.VarLenFeature(tf.int64),
                k.replace("image", "shape"): tf.io.VarLenFeature(tf.int64),
                k.replace("image", "dx"): tf.io.FixedLenFeature((1), float, default_value=-1),
                k.replace("image", "dy"): tf.io.FixedLenFeature((1), float, default_value=-1),
                k.replace("image", "dz"): tf.io.FixedLenFeature((1), float, default_value=-1),
                k.replace("image", "is_seg"): tf.io.FixedLenFeature((1), tf.int64, default_value=-1),
                k.replace("image", "right"): tf.io.FixedLenFeature((1), tf.int64, default_value=-1),
            }
            example = tf.io.parse_example(raw_example, features)
            images.append((k, tf.py_function(
                reconstruct_image,
                (example[k].values, example[k.replace("image", "shape")].values), tf.float32)))

        except tf.errors.InvalidArgumentError as e:
            _logger.error(
                f"Could not pass image with key: {k}. Error: {e}. "
                f"Patient: {patient_details.get('patient_id', 'unknown')}"
            )
            continue

    return (patient_details, images)


def reconstruct_image(image: tf.Tensor, shape: tf.Tensor) -> tf.Tensor:
    """ Reconstructs a 3D image feature from a parsed_example that has undefined shape.

    Args:
        image: An eager tensor of a dimensionless image.
        shape: An eager tensor of the images shape. (i.e can be three of four dimensions)

    Returns:
         A 3D image feature with its original shape reapplied.
    """
    shape = list(filter(lambda x: x != 1, list(shape.numpy())))

    if shape == 3 * [-1]:
        return tf.constant(-1, dtype=tf.float32)

    return tf.cast(tf.reshape(image, shape), tf.float32)

def filter_images(x1, x2, x3, x4, x11, x21, x31, x41, x12, x22, x32, x42, x13, x23, x33, x43, x17, x18,x19) -> tf.Tensor:
# #     """images
# #
# #     Args:
# #         images:
# #     Returns:
# #
# #     """
    images = [x1, x2, x3, x4, x11, x21, x31, x41, x12, x22, x32, x42, x13, x23, x33, x43, x17, x18,x19]

# def filter_images(images) -> tf.Tensor:
    images = list(filter(lambda x: x.shape != [], images))

    if len(images) == 0:
        return -10000 * tf.ones(
            (256,256), dtype=tf.dtypes.float32, name=None
        )

    # TODO: magically pick the best imaging series from all possible.
    image = images[0]

    # TODO: This must magically pick a single slice from 3D (eventually will have to pick 64x64 too.
    image = tf.transpose(image, [2, 1, 0])
    image_z = image.shape[-1]

    image = tf.expand_dims(image[..., image_z//2], -1)

    if image.shape != (256, 256, 1):
        image = tf.image.resize(image, (256, 256))
    return image

def calculate_group_from_results(pCR: Union[int, tf.Tensor], RCB: Union[int, tf.Tensor]) -> Union[int, tf.Tensor]:
    """ Calculates the response group of the patient from their pCR and RCB result.

    Args:
        pCR: Boolean integer (or tf.tensor equivalent) of whether the patient had a complete response to NAC.
        RCB: The residual cancer burden of the patient. [0, 4] (or tf.tensor equivalent)

    Returns:
        The integer, g representing the classification of the patient. 1 <= g <=3, (or tf.tensor equivalent).
    """
    # Remove if statements to make TF's life easier. i.e. make declaration integer/tensor(dtype=int) independent
    return pCR + 2 * ((pCR is None) and (RCB <=2) ) +  3 * ((pCR is None) and RCB > 2 )
    # For integers, is equivalent to:
    # if pCR:
    #     return 1
    # if RCB <= 2:
    #     return 2
    # return 3

def required_imaging_series() -> List[str]:
    """ Returns all the imaging series descriptors that is needed in this model.

    Returns:
        A list of feature names that should be included from imaging TF.examples.
    """
    images = [
        "time1/MRI_DCE/image",
        "time1/SEG_VOI/image",
        "time1/Tissue_SEG/image"
    ]
    return images

def possible_imaging_series_tags() -> List[str]:
    """
    """
    return [f"time1/{x}/image" for x in [
         'FSE_T2_SAG_NO_F', 'FL3D_T1_SAG_CA', '2_FSE_SAGITTAL', 'T2_LEFT_BREAST', 'LT_SAG_T1', 'T2_FSE_UNILAT__SAGITTAL',
         'LT_SAG_FSE_T2___FATSAT', 'FSE_T2_SAG_FS_C', 'FL3D_SAG_UNI_R', 'T1_LEFT_BREAST', 'RT_SAG_T1', 'T1_RIGHT_BREAST',
         'FL3D_T1_SAG_CA_', 'RT_SAG_FSE_T2___FATSAT', 'UNI_SAG_T2_FSE', 'UNI_SAG_T2_FSE_L', 'T2_RIGHT_BREAST',
         'LT_IR_SPGR_SAG', 'RT_IR_SPGR_SAG'
    ]]
