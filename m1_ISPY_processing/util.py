import argparse
from typing import Callable, List, Tuple

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

import constants


def _parse_argv(argv: List[str]) -> Tuple[argparse.Namespace, List[str]]:
    """ Parses CLI parameters, validates and converts them to required format.

    Args:
        argv: A list of arguments from the user's command line.

    Returns:
        A namespace objects of parsed arguments and remaining unknown args. The first contains pipeline specific
        arguments used. The second contains all parameters not explicitly added below and is, in general, meant for
        the underlying Apache Beam pipeline and its configuration (whether using Google DataFlow or not).
    """
    parser = argparse.ArgumentParser(
        prog="ISPY1 Dataset Processing Pipeline.",
        description="This is a pipeline that processes the ISPY1 dataset into TFRecords ready for ML training."
                    "If using Google DataFlow, there are more optional parameters to configure DataFlow itself."
                    "See: https://cloud.google.com/dataflow/docs/guides/specifying-exec-params#setting-other-cloud-dataflow-pipeline-options"
    )

    # CSV files for Patient Pipeline
    parser.add_argument(f"--{constants.PATIENT_OUTCOME_CSV_FILE_KEY.replace('_', '-')}",
                        dest=constants.PATIENT_OUTCOME_CSV_FILE_KEY,
                        help="Path to the Patients' Outcome Data in CSV."
                        )
    parser.add_argument(f"--{constants.PATIENT_CLINICAL_CSV_FILE_KEY.replace('_', '-')}",
                        dest=constants.PATIENT_CLINICAL_CSV_FILE_KEY,
                        help="Path to the Patients' Clinical Data in CSV."
                        )

    known_args, pipeline_args = parser.parse_known_args(argv)
    return known_args, pipeline_args


def run_pipeline(argv: List[str], construct_pipeline: Callable[[argparse.Namespace, beam.Pipeline], None]) -> int:
    """Main program run for data processing."""
    argv, pipeline_arg = _parse_argv(argv)

    # [Complete Pipeline]
    with beam.Pipeline(options=PipelineOptions(pipeline_arg)) as main_pipeline:
        construct_pipeline(argv, main_pipeline)
    return 0
