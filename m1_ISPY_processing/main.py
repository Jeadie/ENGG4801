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
from patient_pipeline import PatientPipeline
from studies_pipeline import studies_pipeline
from util import get_pipeline_argv_from_argv, parse_argv


def main(argv=None) -> int:
    """Main program run for data processing."""
    argv = parse_argv(argv)
    pipeline_arg = get_pipeline_argv_from_argv(argv)

    # [Complete Pipeline]
    # TODO: Make Custom PipelineOptions to access argv in pipeline easier.
    with beam.Pipeline(options=PipelineOptions([f"--{k}={v}" for (k,v) in pipeline_arg.items()])) as main_pipeline:
        output_series_pipeline = series_pipeline(main_pipeline, argv)
        output_patient_pipeline = PatientPipeline(main_pipeline, argv).construct()
        output_studies_pipeline = studies_pipeline(output_series_pipeline, argv)
        merge_and_save_pipeline(output_patient_pipeline, output_studies_pipeline, argv)




if __name__ == "__main__":
    main(sys.argv)