from typing import Dict, Tuple

import pydicom

from . import constants

def construct_metadata_from_DICOM_dictionary(
    dicom: pydicom.Dataset,
) -> Dict[str, object]:
    """ Converts DICOM dictionary tags to a pythonic dictionary.

    Args:
        dicom: An open DICOM dataset.

    Returns:
        The original DICOM dictionary (in the DICOM sense) to a pythonic dictionary.
        The pixel data tag is ignored.
    """
    d = {}
    for element in dicom.values():
        try:
            if element.tag == constants.DICOM_PIXEL_TAG:
                continue

            # Convert RawDataElement
            if type(element.value) in constants.DICOM_SPECIFIC_TYPES:
                # Convert DICOM type to Python type with function lookup table.
                d[element.description()] = constants.DICOM_TYPE_CONVERSION[
                    type(element.value)
                ](element.value)
            else:
                d[element.description()] = element.value
        except (KeyError, TypeError):
            # Just don't include problematic key-values
            continue
    return d


def parse_gcs_path(path: str) -> Tuple[str, str]:
    """ Parses a full google cloud URI, returning its bucket and relative file path

    Ex.
        `gs://bucket_name/prefix/of/file/` -> (`bucket_name`, `prefix/of/file`)

    Args:
        path: A full GCS URI.

    Return: A tuple containing the bucket and relative path of the URI.
    """
    x = path.split("/")
    bucket = x[2]
    prefix = x[3:-1]
    return bucket, "/".join(prefix)