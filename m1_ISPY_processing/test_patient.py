import argparse
import sys

import apache_beam as beam

from pipeline.patient_pipeline import construct
from util import run_pipeline


def construct_patient_test_pipeline(parsed_args: argparse.Namespace, p: beam.Pipeline):
    """ Runs a manual test of the Patient Pipeline. Merely prints each element to STDOUT.
    """
    output_patient_pipeline = construct(p, vars(parsed_args))
    _ = output_patient_pipeline | "Print Results" >> beam.Map(
        lambda x: print(f"Element: {str(x)}")
    )


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_pipeline(sys.argv, construct_patient_test_pipeline)

    else:
        print("Currently, can only run Patient pipeline as test using `--test`.")
