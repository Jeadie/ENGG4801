import argparse
import sys
import tempfile
from functools import reduce
from typing import Dict, List, Tuple

import apache_beam as beam
from apache_beam.pvalue import PCollection
from apache_beam.pipeline import Pipeline
import numpy as np
import pydicom as dicom

from custom_types import Types
from custom_exceptions import (
    DICOMAccessError,
    SeriesConstructionError,
    SeriesMetadataError,
)
from series_filter import SeriesFilter


from google.cloud import bigquery, storage

import constants
import util


def construct(pipeline: Pipeline, settings) -> PCollection:
    """ The patient Pipeline as documented.

    Returns:
         The final PCollection from the patient pipeline.
    """
    f = SeriesFilter(filter_file=settings[constants.SERIES_DESCRIPTION_PATH])
    series_paths = get_all_series(pipeline, settings)
    converted_series = (
        series_paths
        | "Only keep useful Series" >> beam.Filter(lambda x: f.filter_series_path(x))
        | "Parse and convert Series DICOMS" >> beam.Map(lambda x: convert_series(x, f))
        | "Filter out empty directories" >> beam.Filter(lambda x: x is not None)
    )
    return converted_series

def convert_series(series_path: str, filter:SeriesFilter) -> Types.SeriesObj:
    """ Parse a set of DICOMs of a given series, parses out DICOM tags as metadata and
        converts the image to Numpy.

    Args:
        series_path:

    Returns: A tuple of the Series Object for further processing and its
        distribution data for saving.

    """
    try:
        dicoms = get_dicoms(series_path)
        series = construct_series(dicoms, filter)

    except DICOMAccessError:
        error_msg = (
            f"Could not get DICOMS from data source: {series_path}"
        )
    except SeriesMetadataError:
        error_msg = "Could not construct metadata for series."
    except SeriesConstructionError as e:
        error_msg = f"Could not construct series from Dicom. Error: {e}."
    except Exception as e:
        error_msg = f"Unknown Error occurred. Error: {e}. Please rerun."

    else:
        return series

    # On exception, log and return None (which will be filtered out)
    print(
        f"Error occurred when creating series for series: {series_path}. Error: {error_msg}. "
    )
    return None

def construct_series(dicoms: List[Types.SeriesObj], filter: SeriesFilter) -> Types.SeriesObj:
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
        image = np.stack([d[0] for d in dicoms])
        metadata = construct_metadata([d[-1] for d in dicoms], filter)
        return (image, metadata)
    except Exception as e:
        print("ERROR", str(e))
        raise SeriesConstructionError

def construct_metadata(dicom_metadata: List[Dict[str, object]], filter: SeriesFilter) -> Dict[str, object]:
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
    return {
        "time": metadata.get("Clinical Trial Time Point ID", "t-1"),
        "flags": filter.get_series_flags(
            metadata.get("Series Description", "")
        ),
        "Study Instance UID": metadata.get("Study Instance UID", ""),
        "Series Instance UID": metadata.get("Series Instance UID", ""),
        "Clinical Trial Subject ID": metadata.get(
            "Clinical Trial Subject ID", "UNKNOWN"
        ),
        "Spacing Between Slices": metadata.get("Spacing Between Slices", -1),
        "Modality": metadata.get("Modality", ""),
        "Pixel Spacing": metadata.get("Pixel Spacing", -1),
        "Laterality": metadata.get("Laterality", ""),
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
        print("f Error simplifying dicom metadata: {[d.get(k, '') for d in dicom_metadata]}.")
        return ""

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


def get_dicoms(series_path: str) -> List[Types.SeriesObj]:
    """ Gets the DICOMs for a Series.

    Args:
        series_path: URI to the Series directory.

    Returns:
        A list of Series Objects, each with one DICOM in their Series.
    Raises:
        DICOMAccessError: If an error occurs when attempting to get the DICOMs for the particular Series.
    """
    try:
        dicoms = []
        c = storage.Client()
        bucket, prefix = util.parse_gcs_path(series_path)
        # make temp directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Download DICOMS to temp directory
            for dicom in c.list_blobs(bucket, prefix=prefix):
                relative_name = dicom.name.split("/")[-1]
                dicom.download_to_filename(f"{tmp_dir}/{relative_name}")
                d = process_local_DICOM(f"{tmp_dir}/{relative_name}")
                dicoms.append(d)

        return dicoms
    except Exception as e:
        print(f"An error occurred when acquiring Dicom's for {series_path}. Error: {e}. Must rerun to acquire data.")
        raise DICOMAccessError()


def get_all_series(pipeline, settings) -> PCollection[str]:
    """ Gets the path to all the Series in the dataset.

    Returns: A Pcollection of path strings to each Series in the ISPY1 dataset.
    """
    series_path = (
        pipeline
        | "Starting Bigquery" >> beam.Create([constants.BIGQUERY_SERIES_QUERY])
        | "Querying"
        >> beam.FlatMap(
            lambda query: bigquery.Client()
            .query(query)
            .result(page_size=settings.get(constants.SERIES_LIMIT, None))
        )
        | "Convert BQ row to GCS paths"
        >> beam.Map(lambda x: convert_bigquery_row_to_gcs(x, settings))
    )
    return series_path

def convert_bigquery_row_to_gcs(row: bigquery.table.Row, settings) -> str:
    """ Converts a Biquery Row from the ISPY1 Table into the path to its directory in
        Google Cloud Storage.

    Args
        row: A BigQuery row with values: StudyInstanceUID, SeriesInstanceUID

    Returns:
         A Path to the series:
            `gs://<bucket_name>/<dataset_prefix>/StudyInstanceUID/SeriesInstanceUID/)
    """
    return (
        f"{settings[constants.STUDIES_PATH]}"
        f"{row.get(constants.BIGQUERY_STUDY_ID_HEADER)}/"
        f"{row.get(constants.BIGQUERY_SERIES_ID_HEADER)}/"
    )


def construct_series_test_pipeline(parsed_args: argparse.Namespace, p: beam.Pipeline):
    """ Runs a manual test of the Series Pipeline.
    """
    series = construct(p, vars(parsed_args))
    _ = series | "Print Results" >> beam.Map(lambda x: print(f"Element: {str(x)}"))


if __name__ == "__main__":
    if "--test" in sys.argv:
        util.run_pipeline(sys.argv, construct_series_test_pipeline)

    else:
        print("Currently, can only run Series pipeline as test using `--test`.")
