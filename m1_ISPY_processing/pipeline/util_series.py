from functools import reduce
import sys
from typing import Callable, Dict, List, Tuple
import logging

import numpy as np
import pydicom as dicom

from pipeline.custom_types import Types
from pipeline.custom_exceptions import (
    DICOMAccessError,
    SeriesConstructionError,
    SeriesMetadataError,
)
from pipeline.series_filter import SeriesFilter

import pipeline.util as util

_logger = logging.getLogger()


def process_local_DICOM(path: str) -> Tuple[np.array, Dict[str, object]]:
    """ Processes a locally saved, unopen DICOM file.

    Args:
        path: to the DICOM file.
    Returns:
        1. An ndarray of the DICOMs pixel data
        2. The DICOMs data dictionary converted into Pythonic format.
    """
    d = dicom.dcmread(path)
    d.decode()
    metadata = util.construct_metadata_from_DICOM_dictionary(d)
    return (d.pixel_array, metadata)


def convert_series(
    series_path: str, filter: SeriesFilter, get_dicoms_fn: Callable
) -> Types.SeriesObj:
    """ Parse a set of DICOMs of a given series, parses out DICOM tags as metadata and
        converts the image to Numpy.

    Args:
        series_path:

    Returns: A tuple of the Series Object for further processing and its
        distribution data for saving.

    """
    try:
        dicoms = get_dicoms_fn(series_path)
        series = construct_series(dicoms, filter)

    except DICOMAccessError:
        error_msg = f"Could not get DICOMS from data source: {series_path}"
    except SeriesMetadataError:
        error_msg = "Could not construct metadata for series."
    except SeriesConstructionError as e:
        error_msg = f"Could not construct series from Dicom. Error: {e}."
    except Exception as e:
        error_msg = f"Unknown Error occurred. Error: {e}. Please rerun."

    else:
        return series

    # On exception, log and return None (which will be filtered out)
    _logger.error(
        f"Error occurred when creating series for series: {series_path}. Error: {error_msg}. "
    )
    return None


def construct_series(
    dicoms: List[Types.SeriesObj], filter: SeriesFilter
) -> Types.SeriesObj:
    """ Converts a list of parsed DICOM items into a single Series object
    with n+1 dimensional image and metadata merged.

    Args:
        dicoms: A list of imaging objects

    Returns:
        A single SeriesObj
    Raises:
        SeriesConstructionError: If the imaging and metadata for a series could not be constructed.
    """
    try:
        if len(dicoms) == 0:
            return None

        dicoms = sort_dicoms(dicoms)
        image = np.stack([d[0] for d in dicoms])

        metadata = construct_metadata([d[-1] for d in dicoms], filter)
        return (image, metadata)
    except Exception as e:
        _logger.error(f"An error occurred constructing the series. Error: {str(e)}.")
        raise SeriesConstructionError

def sort_dicoms(dicoms: List[Types.SeriesObj]) -> List[Types.SeriesObj]:
    """ Sorts a list of DICOMs based on the z location.

    It first attempts to sort based on the DICOM tag, `Slice Location`.
    Failing this, it sorts on the Z component of `Image Position (Patient)`.

    Args:
        dicoms: A List of DICOM objects that are assumed to be of the same 3D
            image.

    Returns:
          The same list of DICOMs sorted by their z-location.
    """
    # Sort dicoms based on Slice Location tag if they all have slice location
    tag = "Slice Location"
    if True not in [(sys.maxsize == float(x[-1].get(tag, sys.maxsize))) for x in dicoms]:
        dicoms.sort(key=lambda x: float(x[-1].get(tag, sys.maxsize)))
        return dicoms
    else:
        _logger.warning(f"Dicoms do not have Slice Location key.")
        tag = "Image Position (Patient)"
        if True not in [sys.maxsize == float(x[-1].get(tag, (0,0, sys.maxsize))[-1]) for x in dicoms]:
            dicoms.sort(key=lambda x: float(x[-1].get(tag, (0,0, sys.maxsize))[-1]))
        else:
            _logger.warning(f"Dicoms have no z location data. Series is unsorted.")
        return dicoms

def construct_metadata(
    dicom_metadata: List[Dict[str, object]], filter: SeriesFilter
) -> Dict[str, object]:
    """ Constructs the necessary Metadata from a list of DICOM metadata (converted to Pythonic types)

    Args:
        dicom_metadata: A list of metadata from raw DICOMs. Metadata has
            been converted to pythonic types not dicom.valuerep types.

    Returns:
         A formatted metadata dictonary relevant to the Series as a whole.
    """
    # Get the union of keys from all DICOMs
    keys = reduce(
        lambda x, y: x | y,
        [set(dicom_metadata[i].keys()) for i in range(len(dicom_metadata))],
    )



    # Convert list of dictionaries into dictionary of lists (key -> List[values])
    metadata = {k: simplify_values(dicom_metadata, k) for k in keys}

    # Get Dicom specific values into a list. i.e. Dicom z positions, etc
    slice_locations = [float(d.get("Slice Location", sys.maxsize)) for d in dicom_metadata]
    if len(slice_locations) < 2:
        slice_locations = [0,0]

    return {
        "time": metadata.get("Clinical Trial Time Point ID", "t-1"),
        "flags": filter.get_series_flags(metadata.get("Series Description", "")),
        "Study Instance UID": metadata.get("Study Instance UID", ""),
        "Series Instance UID": metadata.get("Series Instance UID", ""),
        "Clinical Trial Subject ID": metadata.get(
            "Clinical Trial Subject ID", "UNKNOWN"
        ),
        "Laterality": metadata.get("Laterality", ""),
        "Spacing Between Slices": metadata.get("Spacing Between Slices", -1),
        "Modality": metadata.get("Modality", ""),
        "Pixel Spacing": metadata.get("Pixel Spacing", (-1, -1)),
        "Image Position (Patient)": metadata.get("Image Position (Patient)", (-1, -1, -1)),
        "Image Orientation (Patient)": metadata.get("Image Orientation (Patient)", (0,0,0,0,0,0)),
        "z_bound" : (slice_locations[0], slice_locations[-1])
    }


def simplify_values(dicom_metadata, k):
    try:
        if type(dicom_metadata[0][k]) == list:
            values = list(set([tuple(d[k]) for d in dicom_metadata if d.get(k)]))
        else:
            values = list(set([d[k] for d in dicom_metadata if d.get(k)]))
        if len(values) == 0:
            return ""
        return values[0]
    except Exception as e:
        _logger.info(
            f"Error simplifying dicom metadata for key, {k} : {[dict(d).get(k, '') for d in dicom_metadata]}."
        )
        return ""
