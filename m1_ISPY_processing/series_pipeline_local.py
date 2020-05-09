import argparse
import logging
import os
import sys
from typing import Dict, List

import apache_beam as beam
from apache_beam.pvalue import PCollection
from apache_beam.pipeline import Pipeline


import constants
from series_filter import SeriesFilter

from custom_exceptions import DICOMAccessError
from custom_types import Types
import util
from util_series import (
    convert_series,
    process_local_DICOM
)

_logger = logging.getLogger()


"""Series Pipeline for files found locally."""


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
        for dicom in list(filter(lambda x: ".dcm" in x, os.listdir(series_path))):
            d = process_local_DICOM(f"{series_path}{dicom}")
            dicoms.append(d)

        return dicoms
    except Exception as e:
        _logger.error(f"An error occurred when acquiring Dicom's for {series_path}. Error: {e}. Must rerun to acquire data.")
        raise DICOMAccessError()

def get_all_series(pipeline: Pipeline, studies_dir: str) -> PCollection[str]:
    """ Gets the path to all the Series in the dataset.

    Returns: A Pcollection of path strings to each Series in the ISPY1 dataset.
    """
    series_path = (
        pipeline
        | beam.Create([studies_dir])
        # Efficiently traverse only two levels of depth, ignoring files.
        | beam.FlatMap(
            lambda path: list(
                map(
                    lambda y: f"{path}{y.name}/",
                    filter(lambda x: x.is_dir(), os.scandir(path)),
                )
            )
        )
        | beam.FlatMap(
            lambda y: list(
                map(
                    lambda x: f"{y}{x.name}/",
                    filter(lambda i: i.is_dir(), list(os.scandir(y))),
                )
            )
        )
    )
    return series_path


def construct(pipeline: Pipeline, settings: Dict[str, object]) -> PCollection:
    """ The patient Pipeline as documented.

    Returns:
         The final PCollection from the patient pipeline.
    """
    filter = SeriesFilter(filter_file=settings[constants.SERIES_DESCRIPTION_PATH])
    series_paths = get_all_series(pipeline, settings)
    converted_series = (
            series_paths
            | "Only keep useful Series" >> beam.Filter(filter.filter_series_path)
            | "Parse and convert Series DICOMS" >> beam.Map(lambda x: convert_series(x, filter, get_dicoms))
            | "Filter out empty directories" >> beam.Filter(lambda x: x is not None)
    )
    return converted_series


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
