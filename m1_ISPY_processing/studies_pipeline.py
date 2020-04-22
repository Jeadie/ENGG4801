import argparse
import sys
from typing import Dict, Tuple

import apache_beam as beam
from apache_beam.pvalue import PCollection

import constants
from custom_types import Types
import util


class StudiesPipeline(object):

    def __init__(self, series_collection: PCollection, argv: Dict[str, object]):
        """ Constructor.
        Args:

            argv: Parsed arguments from CLI.
        """
        self.series = series_collection
        self.settings = argv

    def construct(self) -> PCollection:
        """ The Studies pipeline as documented.

        Returns:
             The final PCollection from the patient pipeline.
        """
        group_by_patients = (
                self.series
                | "Parse out Study ID from each Series Obj" >> beam.Map(lambda x: (x[1].get("Study Instance UID"), x))
                | "Group by Study ID" >> beam.GroupByKey()
                | "Parse out Patient ID" >> beam.Map(self.parse_patient_from_study )
                | "Group by Patient ID" >> beam.GroupByKey()
        )
        return group_by_patients

    def parse_patient_from_study(self, study: Types.StudyObj) -> Tuple[str, Types.StudyObj]:
        """ Turns an element in a PCollection into a keyed, by patient id, element.

        Indexing:
            series= study[1]
            singular_series = series[0]
            singular_series_metadata = singular_series[1]

        Args:
            study: A single Study Object.
        A keyed element Tuple of the form (Patient ID, Study Object).
        """
        return (study[1][0][1].get("Clinical Trial Subject ID"), study)

def construct_studies_test_pipeline(parsed_args: argparse.Namespace, p: beam.Pipeline):
    """ Runs a manual test of the Series Pipeline.
    """
    args = vars(parsed_args)
    series = util.get_series_pipeline(args[constants.STUDIES_PATH])(p, args).construct()
    studies = StudiesPipeline(series, vars(parsed_args)).construct()
    _ = studies | "Print Results" >> beam.Map(lambda x: print(f"Element: {str(x)}"))


if __name__ == '__main__':
    if "--test" in sys.argv:
        util.run_pipeline(sys.argv, construct_studies_test_pipeline)

    else:
        print("Currently, can only run Series pipeline as test using `--test`.")

