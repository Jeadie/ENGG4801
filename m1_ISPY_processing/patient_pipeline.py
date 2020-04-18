import argparse
import csv
import io
import logging
import sys
from typing import Dict, List

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.pvalue import PCollection
from apache_beam.pipeline import Pipeline

import constants
from constants import CSVHeader
import custom_exceptions
from util import get_pipeline_argv_from_argv, parse_argv


class PatientPipeline(object):

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

        Args:
            main_pipeline: A reference to the main pipeline.
            argv: Parsed arguments from CLI.

        Returns:
             The final PCollection from the patient pipeline.
        """
        patients = self.get_all_patients()
        return beam.Map(self.format_patient_metadata, patients)

    def format_patient_metadata(self, patient: List[str]) -> Dict[str, object]:
        """ Parses a single CSV line and extracts and formats the clinical and outcome data.

        Args:
            patient: A single line from the joint CSV of patient clinical and outcome data.

        Return:

        """
        data = dict(filter(lambda x, y: y != "", zip(constants.JOINT_CSV_HEADERS, patient)))
        return {
            "patient_id": int(data[CSVHeader.SUBJECT_ID.value]),
            "demographic_metadata": {
                "age": float(data[CSVHeader.AGE.value]),
                "race": int(data[CSVHeader.RACE.value]),
            },
            "clinical": {
                "ERpos": int(data.get(CSVHeader.ERpos.value, -1)),
                "Pgpos": int(data.get(CSVHeader.PgRpos.value, -1)),
                "HRpos": int(data.get(CSVHeader.HRPos.value, -1)),
                "HER_two_status": int(data.get(CSVHeader.HER2STATUS.value)),
                "three_level_HER": int(data.get(CSVHeader.TRIPLE_LEVEL_HER.value)),
                "Bilateral": int(data.get(CSVHeader.BILATERAL_CANCER.value)),
                "Laterality": int(data.get(CSVHeader.LATERALITY.value)),
            },
            "LD": [int(data.get(x.value, -1)) for x in
                   [CSVHeader.LD_BASELINE, CSVHeader.LD_POST_AC, CSVHeader.LD_INTER_REG, CSVHeader.LD_PRE_SURGERY]]
            "outcome": {
                "Sstat": int(data.get(CSVHeader.SURVIVAL_INDICATOR)),
                "survival_duration": int(data.get(CSVHeader.SURVIVAL_DURATION)),
                "rfs_ind": int(data.get(CSVHeader.RECURRENCE_FREE_INDICATOR)),
                "rfs_duration": int(data.get(CSVHeader.RECURRENCE_FREE_DURATION)),
                "pCR": int(data.get(CSVHeader.PATHOLOGICAL_COMPLETE_RESPONSE, -1)),
                "RCB": int(data.get(CSVHeader.RESIDUAL_CANCER_BURDEN_CLASS, -1)),
            }
        }


    # TODO: This seems sloppy.
    def get_all_patients(self) -> PCollection:
        """ Parses the two CSV files, outcomes and clinical, and merges them based on PatientID.

        It is assumed the two CSVs are ordered, ascendingly, by PatientID.

        Returns:
            A PCollection of unparsed CSV data merged from the two CSV pages.
        """

        # Both of these create PCollections with iterators that __next__() -> List[str]
        outcomes_data = beam.Map(lambda x: (x[0], x[1:]),
                                 self.load_csv(self.settings[constants.PATIENT_OUTCOME_CSV_FILE_KEY]))
        clinical_data = beam.Map(lambda x: (x[0], x[1:]),
                                 self.load_csv(self.settings[constants.PATIENT_CLINICAL_CSV_FILE_KEY]))
        return (
                {"outcomes": outcomes_data, "clinical": clinical_data} | beam.CoGroupByKey()
                | beam.Map(lambda patient_id, maps: ",".join([patient_id] + maps["outcomes"] + maps["clinical"]))
        )

    def load_csv(self, csv_path: str) -> PCollection:
        """ Loads a CSV from a file.

        Args:
            csv_path: Path to the CSV to load.

        Returns:
            A PCollection whereby each element is a row from the CSV with type, List[str].
        """
        return (self.pipeline
                | beam.io.ReadFromText(csv_path)
                | beam.FlatMap(lambda x: x.split(constants.CSV_DELIMETER))
                )


def test_patient_pipeline(argv):
    """ Runs a manual test of the Patient Pipeline. Merely prints each element to STDOUT.

    :param argv:
    :return:
    """
    argv = parse_argv(argv)
    pipeline_arg = get_pipeline_argv_from_argv(argv)

    with beam.Pipeline(options=PipelineOptions([f"--{k}={v}" for (k,v) in pipeline_arg.items()])) as test_pipeline:
        output_patient_pipeline = PatientPipeline(test_pipeline, argv).construct()
        beam.map(lambda x: print(f"Element: {str(x)}"), output_patient_pipeline)

if __name__ == '__main__':
    if "--test" in sys.argv:
        test_patient_pipeline(sys.argv)
    else:
        print("Currently, can only run Patient pipeline as test using `--test`.")
        