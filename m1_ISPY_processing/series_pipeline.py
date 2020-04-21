import argparse
from functools import reduce
import os
import sys
import tempfile
from typing import Dict, List, Tuple

import apache_beam as beam
from apache_beam.pvalue import PCollection
from apache_beam.pipeline import Pipeline
from google.cloud import bigquery, storage
import numpy as np
import pydicom as dicom

import constants
import util

SeriesObj = Tuple[np.array, Dict[str, object]]


# TODO: Convert to two separate Series Pipelines: One for local files, one for those on GCS.
class SeriesPipeline(object):

    def __init__(self, main_pipeline: Pipeline, argv: Dict[str, object]):
        """ Constructor.
        Args:
            main_pipeline: A reference to the main pipeline.
            argv: Parsed arguments from CLI.
        """
        self.pipeline = main_pipeline
        self.settings = argv

    def construct(self) -> PCollection:
        """ The patient Pipeline as documented.

        Returns:
             The final PCollection from the patient pipeline.
        """
        series_paths = self.get_all_series()
        converted_series = (
                series_paths
                | "Parse and convert Series DICOMS" >> beam.Map(self.convert_series)
        )
        _ = (
                converted_series
                | "Get Metadata from Series" >> beam.Map(lambda x: x[0])
            # | "Save Metadata to Disk" >> self.process_series_distribution
        )
        return (
                converted_series
                | "Get Series Object from Series" >> beam.Map(lambda x: x[-1])
        )

    def process_series_distribution(self, dist: Dict[str, str]) -> None:
        """ Saves a series distribution to CSV.

        :return:
        """
        return None

    def convert_series(self, series_path: str) -> Tuple[SeriesObj, int]:
        """ Parse a set of DICOMs of a given series, parses out DICOM tags as metadata and converts the image to Numpy.

        Args:
            series_path:

        Returns: A tuple of the Series Object for further processing and its distribution data for saving.

        """
        dicoms = []
        # TODO Two classes as mentioned above
        # GCS Storage
        if constants.GCS_PREFIX in series_path:
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
        # local_storage
        else:
            # TODO: I think this should recurse multiple levels
            for dicom in list(filter(lambda x: ".dcm" in x, os.listdir(series_path))):
                d = self.process_local_DICOM(f"{series_path}{dicom}")
                dicoms.append(d)

        meta = self.construct_metadata([d[-1] for d in dicoms])
        series = self.construct_series(dicoms)
        return (
            meta,
            series
        )

    def construct_series(self, dicoms: List[SeriesObj]) -> SeriesObj:
        """ Converts a list of parsed DICOM items into a single Series object with n+1 dimensional image and metadata merged.

        Args:
            dicoms: A list of imaging objects

        Returns:
            A single SeriesObj
        """
        image = np.stack([d[0] for d in dicoms])

        # TODO: Merge metadata instead of just mapping individually.
        metadata = self.construct_metadata([d[-1] for d in dicoms])
        return (
            image,
            metadata
        )

    def construct_metadata(self, dicom_metadata: List[Dict[str, object]]) -> Dict[str, object]:
        """

        :param dicom_metadata:
        :return:
        """
        # Get the union of keys from all DICOMs
        keys = reduce(lambda x, y: x | y, [set(dicom_metadata[i].keys()) for i in range(len(dicom_metadata))])

        # Convert list of dictionaries into dictionary of lists (key -> List[values])
        metadata = {k: [d[k] for d in dicom_metadata if d.get(k)] for k in keys}
        return metadata

    def process_local_DICOM(self, path: str) -> Tuple[np.array, Dict[str, object]]:
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
        return (
            d.pixel_array,
            metadata
        )

    # TODO: We may need to filter by bad Series (Segmentations, for example)
    def get_all_series(self) -> PCollection[str]:
        """ Gets the path to all the Series in the dataset.

        Returns: A Pcollection of path strings to each Series in the ISPY1 dataset.
        """
        studies_dir = self.settings[constants.STUDIES_PATH]
        if constants.GCS_PREFIX in studies_dir:
            series_path = (
                    self.pipeline
                    | "Starting Bigquery" >> beam.Create([constants.BIGQUERY_SERIES_QUERY])
                    | "Querying" >> beam.FlatMap(lambda query: bigquery.Client().query(query).result(
                max_results=self.settings.get(constants.SERIES_LIMIT, None)))
                    | "Convert BQ row to GCS paths" >> beam.Map(self.convert_bigquery_row_to_gcs)
            )
        else:
            series_path = (
                    self.pipeline
                    | beam.Create([studies_dir])
                    # Efficiently traverse only two levels of depth, ignoring files.
                    # TODO: Efficiently avoids DICOms, but is ugly and unclean
                    | beam.FlatMap(
                lambda path: list(map(lambda y: f"{path}{y.name}/", filter(lambda x: x.is_dir(), os.scandir(path)))))
                    | beam.FlatMap(
                lambda y: list(map(lambda x: f"{y}{x.name}/", filter(lambda i: i.is_dir(), list(os.scandir(y))))))
            )

        return series_path

    def convert_bigquery_row_to_gcs(self, row: bigquery.table.Row) -> str:
        """ Converts a Biquery Row from the ISPY1 Table into the path to its directory in Google Cloud Storage.

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
    series = SeriesPipeline(p, vars(parsed_args)).construct()
    _ = series | "Print Results" >> beam.Map(lambda x: print(f"Element: {str(x)}"))


if __name__ == '__main__':
    if "--test" in sys.argv:
        util.run_pipeline(sys.argv, construct_series_test_pipeline)

    else:
        print("Currently, can only run Series pipeline as test using `--test`.")
