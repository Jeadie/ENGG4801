import argparse
import sys
import tempfile
from typing import Dict, List

import apache_beam as beam
from apache_beam.pvalue import PCollection
from google.cloud import bigquery, storage

import constants
from series_pipeline_base import BaseSeriesPipeline
from custom_exceptions import DICOMAccessError
from custom_types import Types
import util


class GCSSeriesPipeline(BaseSeriesPipeline):
    """Series Pipeline for files stored in GCS and thus require Google's Bigquery
        to find all series.
    """

    def process_series_distribution(self, dist: Dict[str, str]) -> None:
        """ Saves a series distribution to CSV.
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
            c = storage.Client()
            bucket, prefix = util.parse_gcs_path(series_path)
            # make temp directory
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Download DICOMS to temp directory
                for dicom in c.list_blobs(bucket, prefix=prefix):
                    relative_name = dicom.name.split("/")[-1]
                    dicom.download_to_filename(f"{tmp_dir}/{relative_name}")
                    d = self.process_local_DICOM(f"{tmp_dir}/{relative_name}")
                    dicoms.append(d)

            return dicoms
        except Exception as e:
            print(f"An error occurred when acquiring Dicom's for {series_path}. Error: {e}. Must rerun to acquire data.")
            raise DICOMAccessError()

    def get_all_series(self) -> PCollection[str]:
        """ Gets the path to all the Series in the dataset.

        Returns: A Pcollection of path strings to each Series in the ISPY1 dataset.
        """
        series_path = (
            self.pipeline
            | "Starting Bigquery" >> beam.Create([constants.BIGQUERY_SERIES_QUERY])
            | "Querying"
            >> beam.FlatMap(
                lambda query: bigquery.Client()
                .query(query)
                .result(max_results=self.settings.get(constants.SERIES_LIMIT, None))
            )
            | "Convert BQ row to GCS paths"
            >> beam.Map(self.convert_bigquery_row_to_gcs)
        )
        return series_path

    def convert_bigquery_row_to_gcs(self, row: bigquery.table.Row) -> str:
        """ Converts a Biquery Row from the ISPY1 Table into the path to its directory in
            Google Cloud Storage.

        Args
            row: A BigQuery row with values: StudyInstanceUID, SeriesInstanceUID

        Returns:
             A Path to the series:
                `gs://<bucket_name>/<dataset_prefix>/StudyInstanceUID/SeriesInstanceUID/)
        """
        return (
            f"{self.settings[constants.STUDIES_PATH]}"
            f"{row.get(constants.BIGQUERY_STUDY_ID_HEADER)}/"
            f"{row.get(constants.BIGQUERY_SERIES_ID_HEADER)}/"
        )


def construct_series_test_pipeline(parsed_args: argparse.Namespace, p: beam.Pipeline):
    """ Runs a manual test of the Series Pipeline.
    """
    series = GCSSeriesPipeline(p, vars(parsed_args)).construct()
    _ = series | "Print Results" >> beam.Map(lambda x: print(f"Element: {str(x)}"))


if __name__ == "__main__":
    if "--test" in sys.argv:
        util.run_pipeline(sys.argv, construct_series_test_pipeline)

    else:
        print("Currently, can only run Series pipeline as test using `--test`.")
