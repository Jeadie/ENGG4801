from __future__ import absolute_import

import argparse
import logging
import sys
from typing import Dict, List

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

import constants
import custom_exceptions
from merge_and_save_pipeline import merge_and_save_pipeline
from series_pipeline import series_pipeline
from patient_pipeline import patient_pipeline
from studies_pipeline import studies_pipeline


def parse_argv(argv:List[str]) -> Dict[str: object]:
    """ Parses CLI parameters, validates and converts them to required format.

    Args:
        argv: A list of arguments from the user's command line.

    Returns:
        A dictionary mapping flags to values.
            i.e. --key=value -> {"key": "value"}
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file to process.")
    for k, help in constants.PIPELINE_CLI_ARGUMENTS:
        parser.add_argument(f"--{k}", dest=k, help=help)

    known_args = parser.parse_known_args(argv)
    return dict(known_args)

def get_pipeline_argv_from_argv(argv: Dict[str, object]) -> Dict[str, object]:
    """ Collates the pipeline arguments from the CLI argv.

    Args:
        argv: CLI argv

    Returns:
        A key, value mapping of arguments to pass into the pipeline options itself.
    """
    pipeline_argv = [(k, argv.get(k, None)) for k in constants.PIPELINE_CLI_ARGUMENTS]
    pipeline_argv = list(filter(lambda x: x[-1] != None, pipeline_argv))
    return dict(pipeline_argv)

def main(argv=None) -> int:
    """Main program run for data processing."""
    argv = parse_argv(argv)
    pipeline_arg = get_pipeline_argv_from_argv(argv)

    # [Complete Pipeline]
    # TODO: Make Custom PipelineOptions to access argv in pipeline easier.
    with beam.Pipeline(options=PipelineOptions([f"--{k}={v}" for (k,v) in pipeline_arg.items()])) as main_pipeline:
        output_series_pipeline = series_pipeline(main_pipeline, argv)
        output_patient_pipeline = patient_pipeline(main_pipeline, argv)
        output_studies_pipeline = studies_pipeline(output_series_pipeline, argv)
        merge_and_save_pipeline(output_patient_pipeline, output_studies_pipeline, argv)




if __name__ == "__main__":
    main(sys.argv)