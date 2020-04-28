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
import constants
from series_filter import SeriesFilter
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
        self.filter = SeriesFilter()

    def construct(self) -> PCollection:
        """ The patient Pipeline as documented.

        Returns:
             The final PCollection from the patient pipeline.
        """
        series_paths = self.get_all_series()
        converted_series = (
            series_paths
            | "Only keep useful Series" >> beam.Filter(self.filter.filter_series_path)
            | "Parse and convert Series DICOMS" >> beam.Map(self.convert_series)
            | "Filter out empty directories" >> beam.Filter(lambda x: x is not None)
        )
        _ = (
            converted_series
            | "Get Metadata from Series" >> beam.Map(lambda x: x[0])
            # | "Save Metadata to Disk" >> self.process_series_distribution
        )
        return converted_series | "Get Series Object from Series" >> beam.Map(
            lambda x: x[-1]
        )

    # TODO: This should be done in this class
    def process_series_distribution(self, dist: Dict[str, str]) -> None:
        """ Saves a series distribution to CSV.

        :return:
        """
        return None

    def convert_series(self, series_path: str) -> Tuple[Types.SeriesObj, int]:
        """ Parse a set of DICOMs of a given series, parses out DICOM tags as metadata and
            converts the image to Numpy.

        Args:
            series_path:

        Returns: A tuple of the Series Object for further processing and its
            distribution data for saving.

        """
        try:
            dicoms = self.get_dicoms(series_path)
            if len(dicoms) == 0:
                return None

            meta = self.construct_metadata([d[-1] for d in dicoms])
            series = self.construct_series(dicoms)

        except DICOMAccessError:
            error_msg = (
                f"Could not get DICOMS from data source: "
                f"{self.settings.get(constants.STUDIES_PATH, 'Failed to log path to Data source.')}"
            )
        except SeriesMetadataError:
            error_msg = "Could not construct metadata for series."
        except SeriesConstructionError as e:
            error_msg = f"Could not construct series from Dicom. Error: {e}."
        else:
            return (meta, series)

        # On exception, log and return None (which will be filtered out)
        print(
            f"Error occurred when creating series for series: {series_path}. Error: {error_msg}. "
        )
        return None

    def get_dicoms(self, series_path: str) -> List[SeriesObj]:
        """ Gets the DICOMs for a Series.

        Args:
            series_path: URI to the Series directory.

        Returns:
            A list of Series Objects, each with one DICOM in their Series.
        """
        raise NotImplementedError(
            "Base Class, `BaseSeriesPipeline` does not implement `convert_series`"
        )

    def construct_series(self, dicoms: List[Types.SeriesObj]) -> SeriesObj:
        """ Converts a list of parsed DICOM items into a single Series object
        with n+1 dimensional image and metadata merged.

        Args:
            dicoms: A list of imaging objects

        Returns:
            A single SeriesObj
        """
        image = np.stack([d[0] for d in dicoms])
        metadata = self.construct_metadata([d[-1] for d in dicoms])
        return (image, metadata)

    def construct_metadata(
        self, dicom_metadata: List[Dict[str, object]]
    ) -> Dict[str, object]:
        """

        :param dicom_metadata:
        :return:
        """
        # Get the union of keys from all DICOMs
        keys = reduce(
            lambda x, y: x | y,
            [set(dicom_metadata[i].keys()) for i in range(len(dicom_metadata))],
        )

        # Convert list of dictionaries into dictionary of lists (key -> List[values])
        metadata = {k: self.simplify_values(dicom_metadata, k) for k in keys}
        return {
            "time": metadata.get("Clinical Trial Time Point ID", "t-1"),
            "flags": self.filter.get_series_flags(
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
        # return metadata

    def simplify_values(self, dicom_metadata, k):
        if type(dicom_metadata[0][k]) == list:
            values = list(set([tuple(d[k]) for d in dicom_metadata if d.get(k)]))
        else:
            values = list(set([d[k] for d in dicom_metadata if d.get(k)]))
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
        return (d.pixel_array, metadata)

    def get_all_series(self) -> PCollection[str]:
        """ Gets the path to all the Series in the dataset.

        Returns: A Pcollection of path strings to each Series in the ISPY1 dataset.
        """
        raise NotImplementedError(
            "Base Class, `BaseSeriesPipeline` does not implement `get_all_series`"
        )
