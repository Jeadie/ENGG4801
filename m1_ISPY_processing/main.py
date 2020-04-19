import argparse
import sys
from typing import List

import apache_beam as beam

from merge_and_save_pipeline import merge_and_save_pipeline
from series_pipeline import SeriesPipeline
from patient_pipeline import PatientPipeline
from studies_pipeline import studies_pipeline
from util import get_pipeline_argv_from_argv, parse_argv, run_pipeline


def main(argv: List[str]) -> int:
    """Main program run for data processing."""
    return run_pipeline(argv, construct_main_pipeline)

def construct_main_pipeline(parsed_args: argparse.Namespace, p: beam.Pipeline) -> None:
    """ Responsible for constructing the main pipeline for processing ISPY1.

    Args:
        parsed_args: CLI arguments parsed and validated.
    :param p: A pipeline, preconfigured with pipeline options.
    """
    output_series_pipeline = SeriesPipeline(p, parsed_args).construct()
    output_patient_pipeline = PatientPipeline(p, parsed_args).construct()
    output_studies_pipeline = studies_pipeline(output_series_pipeline, parsed_args)
    merge_and_save_pipeline(output_patient_pipeline, output_studies_pipeline, parsed_args)

if __name__ == "__main__":
    main(sys.argv)
