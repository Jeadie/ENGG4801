import argparse
import sys

import apache_beam as beam

from series_pipeline_gcs import construct
import util


def construct_series_test_pipeline(parsed_args: argparse.Namespace, p: beam.Pipeline):
    """ Runs a manual test of the Series Pipeline.
    """
    series = construct(p, vars(parsed_args))
    _ = series | "Print Results" >> beam.Map(lambda x: print(f"Element: {str(x)}"))


if __name__ == "__main__":
    if "--test" in sys.argv:
        util.run_pipeline(sys.argv, construct_series_test_pipeline)

    else:
        print("Currently, can only run Series pipeline as test using `--test`.")
