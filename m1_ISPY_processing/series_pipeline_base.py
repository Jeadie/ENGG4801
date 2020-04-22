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


class BaseSeriesPipeline(object):

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

    # TODO: This should be done in this class
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
        dicoms = self.get_dicoms(series_path)
        meta = self.construct_metadata([d[-1] for d in dicoms])
        series = self.construct_series(dicoms)
        return (
            meta,
            series
        )


    def get_dicoms(self, series_path: str) -> List[SeriesObj]:
        """ Gets the DICOMs for a Series.

        Args:
            series_path: URI to the Series directory.

        Returns:
            A list of Series Objects, each with one DICOM in their Series.
        """
        raise NotImplementedError("Base Class, `BaseSeriesPipeline` does not implement `convert_series`")


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
        metadata = {k: self.simplify_values(dicom_metadata, k) for k in keys}
        return metadata

    def simplify_values(self, dicom_metadata, k):
        if type(dicom_metadata[0][k]) == list:
            values = list(set([tuple(d[k]) for d in dicom_metadata if d.get(k)]))
        else:
            values = list(set([d[k] for d in dicom_metadata if d.get(k)]))
        if len(values) > 1:
            print(f"multiple uniq values for key: {k}, values: {values}")
        if len(values) == 0:
            return ""
        return values[0]

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
        raise NotImplementedError("Base Class, `BaseSeriesPipeline` does not implement `get_all_series`")


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

