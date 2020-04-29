import argparse
import os
import sys
from typing import Dict, List

import apache_beam as beam
from apache_beam.pvalue import PCollection


import constants
from series_pipeline_base import BaseSeriesPipeline
from custom_exceptions import DICOMAccessError
from custom_types import Types
import util


class LocalSeriesPipeline(BaseSeriesPipeline):
    """Series Pipeline for files found locally."""

    def process_series_distribution(self, dist: Dict[str, str]) -> None:
        """ Saves a series distribution to CSV.

        :return:
        """
        return None

    def get_dicoms(self, series_path: str) -> List[Types.SeriesObj]:
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
                d = self.process_local_DICOM(f"{series_path}{dicom}")
                dicoms.append(d)

            return dicoms
        except Exception as e:
            print(f"An error occurred when acquiring Dicom's for {series_path}. Error: {e}. Must rerun to acquire data.")
            raise DICOMAccessError()

    def get_all_series(self) -> PCollection[str]:
        """ Gets the path to all the Series in the dataset.

        Returns: A Pcollection of path strings to each Series in the ISPY1 dataset.
        """
        studies_dir = self.settings[constants.STUDIES_PATH]
        series_path = (
            self.pipeline
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


def construct_series_test_pipeline(parsed_args: argparse.Namespace, p: beam.Pipeline):
    """ Runs a manual test of the Series Pipeline.
    """
    series = LocalSeriesPipeline(p, vars(parsed_args)).construct()
    _ = series | "Print Results" >> beam.Map(lambda x: print(f"Element: {str(x)}"))


if __name__ == "__main__":
    if "--test" in sys.argv:
        util.run_pipeline(sys.argv, construct_series_test_pipeline)

    else:
        print("Currently, can only run Series pipeline as test using `--test`.")
