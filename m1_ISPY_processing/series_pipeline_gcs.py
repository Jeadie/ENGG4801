import argparse
import logging
import sys
import tempfile
from typing import List

import apache_beam as beam
from apache_beam.pvalue import PCollection
from apache_beam.pipeline import Pipeline

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
from util_series import (
    convert_series,
    process_local_DICOM
)

_logger = logging.getLogger()

def construct(pipeline: Pipeline, settings) -> PCollection:
    """ The patient Pipeline as documented.

    Returns:
         The final PCollection from the patient pipeline.
    """
    f = SeriesFilter(filter_file=settings[constants.SERIES_DESCRIPTION_PATH])
    if settings.get("SPECIFIC_GCS", None):
        series_paths = (
          pipeline 
          | beam.Create(settings["SPECIFIC_GCS"])
        )

    else:
        series_paths = get_all_series(pipeline, settings, f)
        series_paths = (series_paths | "Only keep useful Series" >> beam.Filter(lambda x: f.filter_series_path(x)))

    converted_series = (
        series_paths
        | "Parse and convert Series DICOMS" >> beam.Map(lambda x: convert_series(x, f, get_dicoms))
        | "Filter out empty directories" >> beam.Filter(lambda x: x is not None)
    )
    return converted_series


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
        _logger.error(f"An error occurred when acquiring Dicom's for {series_path}. Error: {e}. Must rerun to acquire data.")
        raise DICOMAccessError()


def get_all_series(pipeline, settings, _filter) -> PCollection[str]:
    """ Gets the path to all the Series in the dataset.

    Returns: A Pcollection of path strings to each Series in the ISPY1 dataset.
    """

    if settings.get("SPECIFIC_SERIES", None):
         query = (
            f"SELECT DISTINCT StudyInstanceUID, SeriesInstanceUID "
            f"FROM `chc-tcia.ispy1.ispy1` WHERE SeriesInstanceUID IN {tuple(settings.get('SPECIFIC_SERIES'))}"
            f"GROUP BY StudyInstanceUID, SeriesInstanceUID"
        )
    else:
        query = (
        f"SELECT DISTINCT StudyInstanceUID, SeriesInstanceUID "
        f"FROM `chc-tcia.ispy1.ispy1` GROUP BY StudyInstanceUID, SeriesInstanceUID"
    )

    series_path = (
        pipeline
        | "Starting Bigquery" >> beam.Create([query])
        | "Querying"
        >> beam.FlatMap(
            lambda q: bigquery.Client()
            .query(q)
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
    _ = series | "Print Results" >> beam.Map(lambda x: _logger.info(f"Element: {str(x)}"))


if __name__ == "__main__":
    if "--test" in sys.argv:
        util.run_pipeline(sys.argv, construct_series_test_pipeline)

    else:
        _logger.info("Currently, can only run Series pipeline as test using `--test`.")
