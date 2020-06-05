import argparse
import sys

import apache_beam as beam

import pipeline.constants as constants
from pipeline.studies_pipeline import construct
import util


def construct_studies_test_pipeline(parsed_args: argparse.Namespace, p: beam.Pipeline):
    """ Runs a manual test of the Series Pipeline.
    """
    args = vars(parsed_args)
    series = util.get_series_pipeline(args[constants.STUDIES_PATH])(p, args)
    studies = construct(series)

    _ = studies | "Print Results" >> beam.Map(lambda x: print(f"Element: {str(x)}"))


if __name__ == "__main__":
    if "--test" in sys.argv:
        util.run_pipeline(sys.argv, construct_studies_test_pipeline)

    else:
        print("Currently, can only run Series pipeline as test using `--test`.")
