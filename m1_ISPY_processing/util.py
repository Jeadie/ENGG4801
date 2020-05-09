import argparse
import os
from typing import Callable, Dict, List, Tuple

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import pydicom
import constants


def _parse_argv(argv: List[str]) -> Tuple[argparse.Namespace, List[str]]:
    """ Parses CLI parameters, validates and converts them to required format.

    Args:
        argv: A list of arguments from the user's command line.

    Returns:
        A namespace objects of parsed arguments and remaining unknown args. The first
        contains pipeline specific arguments used. The second contains all parameters
        not explicitly added below and is, in general, meant for the underlying Apache
        Beam pipeline and its configuration (whether using Google DataFlow or not).
    """
    parser = argparse.ArgumentParser(
        prog="ISPY1 Dataset Processing Pipeline.",
        description="This is a pipeline that processes the ISPY1 dataset into TFRecords ready for ML training."
        "If using Google DataFlow, there are more optional parameters to configure DataFlow itself."
        "See: https://cloud.google.com/dataflow/docs/guides/specifying-exec-params#setting-other-cloud-dataflow-pipeline-options",
    )

    # CSV files for Patient Pipeline
    parser.add_argument(
        f"--{constants.PATIENT_OUTCOME_CSV_FILE_KEY.replace('_', '-')}",
        dest=constants.PATIENT_OUTCOME_CSV_FILE_KEY,
        help="Path to the Patients' Outcome Data in CSV.",
    )
    parser.add_argument(
        f"--{constants.PATIENT_CLINICAL_CSV_FILE_KEY.replace('_', '-')}",
        dest=constants.PATIENT_CLINICAL_CSV_FILE_KEY,
        help="Path to the Patients' Clinical Data in CSV.",
    )

    # Series GCS path
    parser.add_argument(
        f"--{constants.STUDIES_PATH.replace('_', '-')}",
        dest=constants.STUDIES_PATH,
        help="Directory of folder containing studies and series.",
    )
    parser.add_argument(
        f"--{constants.SERIES_LIMIT.replace('_', '-')}",
        dest=constants.SERIES_LIMIT,
        type=int,
        help="Number of series to process in total.",
        default=None,  # Bigquery specific value
    )
    parser.add_argument(
        f"--{constants.TFRECORD_NAME.replace('_', '-')}",
        dest=constants.TFRECORD_NAME,
        help=(
            f"Name and directory to save the TFRecord as. TFRecord will be saved "
            f"as `TFRECORD_NAME-SHARD.tfrecords`"
        ),
        default=os.getcwd(),
    )
    parser.add_argument(
        f"--{constants.NUM_TFRECORD_SHARDS.replace('_', '-')}",
        dest=constants.NUM_TFRECORD_SHARDS,
        type=int,
        help="Number of shards to save the TFRecord into.",
        default=1,
    )
    parser.add_argument(
        f"--{constants.SERIES_DESCRIPTION_PATH.replace('_', '-')}",
        dest=constants.SERIES_DESCRIPTION_PATH,
        type=str,
        help="The path to the series description CSV.",
        default="ISPY1_MetaData.csv",
    )
    known_args, pipeline_args = parser.parse_known_args(argv)
    return known_args, pipeline_args


def run_pipeline(
    argv: List[str],
    construct_pipeline: Callable[[argparse.Namespace, beam.Pipeline], None],
) -> int:
    """ Runs a arbitrary pipeline.

    Args:
        argv: User CLI args, unpassed (e.g sys.args)
        construct_pipeline: A function that constructs the pipeline to run.

    Returns:
        0 if no errors occured.
    """
    argv, pipeline_arg = _parse_argv(argv)

    # [Complete Pipeline]
    with beam.Pipeline(options=PipelineOptions(pipeline_arg)) as main_pipeline:
        construct_pipeline(argv, main_pipeline)
    return 0


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


# def get_series_pipeline(studies_path: str) -> type:
#     """ Gets the appropriate series pipeline given the location of its studies.
#
#     Args:
#         studies_path: Path the the studies.
#
#     Returns:
#         The type of BaseSeriesPipeline to use, either GCSSeriesPipeline or
#         LocalSeriesPipeline.
#     """
#     if constants.GCS_PREFIX in studies_path:
#         return GCSSeriesPipeline
#     else:
#         return LocalSeriesPipeline
