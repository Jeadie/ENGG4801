import argparse
import sys
from typing import List

import apache_beam as beam

import constants
from merge_and_save_pipeline import MergeSavePipeline
from patient_pipeline import PatientPipeline
from studies_pipeline import StudiesPipeline
from util import get_series_pipeline, run_pipeline


def main(argv: List[str]) -> int:
    """Main program run for data processing."""
    return run_pipeline(argv, construct_main_pipeline)


def construct_main_pipeline(parsed_args: argparse.Namespace, p: beam.Pipeline) -> None:
    """ Responsible for constructing the main pipeline for processing ISPY1.

    Args:
        parsed_args: CLI arguments parsed and validated.
    :param p: A pipeline, preconfigured with pipeline options.
    """
    args = vars(parsed_args)

    # Dynamically get proper series pipeline based on local/gcs studies path.
    output_series = get_series_pipeline(args.get(constants.STUDIES_PATH))(
        p, args
    ).construct()
    output_patient = PatientPipeline(p, args).construct()
    output_studies = StudiesPipeline(output_series, args).construct()
    MergeSavePipeline(output_patient, output_studies, args).construct()


if __name__ == "__main__":
    main(sys.argv)
