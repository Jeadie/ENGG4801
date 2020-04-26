import argparse
import sys
from typing import Dict, Tuple

import apache_beam as beam
import numpy as np

import constants
import util

SeriesObj = Tuple[np.array, Dict[str, object]]


def construct_series_test_pipeline(parsed_args: argparse.Namespace, p: beam.Pipeline):
    """ Runs a manual test of the Series Pipeline.
    """
    args = vars(parsed_args)
    series = util.get_series_pipeline(args[constants.STUDIES_PATH])(p, args).construct()
    _ = series | "Print Results" >> beam.Map(lambda x: print(f"Element: {str(x)}"))


if __name__ == "__main__":
    if "--test" in sys.argv:
        util.run_pipeline(sys.argv, construct_series_test_pipeline)

    else:
        print("Currently, can only run Series pipeline as test using `--test`.")
