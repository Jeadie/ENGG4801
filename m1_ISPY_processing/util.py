import argparse
from typing import Dict, List

import constants


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
    parser.add_argument(f"--{constants.PATIENT_OUTCOME_CSV_FILE_KEY}", dest=constants.PATIENT_OUTCOME_CSV_FILE_KEY, help="Path to the Patients' Outcome Data in CSV.")
    parser.add_argument(f"--{constants.PATIENT_CLINICAL_CSV_FILE_KEY}", dest=constants.PATIENT_CLINICAL_CSV_FILE_KEY, help="Path to the Patients' Clinical Data in CSV.")

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