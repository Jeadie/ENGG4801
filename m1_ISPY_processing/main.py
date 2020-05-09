import argparse
import os
import sys
from typing import List

import apache_beam as beam
from google.cloud import storage

import constants
from merge_and_save_pipeline import MergeSavePipeline
from patient_pipeline import PatientPipeline
from series_pipeline_gcs import construct as series_construct
from series_filter import SeriesFilter
from studies_pipeline import StudiesPipeline
from util import run_pipeline


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
    output_series = series_construct(p, args)
    output_patient = PatientPipeline(p, args).construct()
    output_studies = StudiesPipeline(output_series, args).construct()
    MergeSavePipeline(output_patient, output_studies, args).construct()

def construct_sequence_pipeline(files, extended_args) -> None:
    """ Responsible for constructing the main pipeline for processing ISPY1.

    Args:
        parsed_args: CLI arguments parsed and validated.
    :param p: A pipeline, preconfigured with pipeline options.
    """

    def constructor(parsed_args: argparse.Namespace, p: beam.Pipeline) -> None:
        args = vars(parsed_args)
        args = {**args, **extended_args}
        # Dynamically get proper series pipeline based on local/gcs studies path.
        output_series = series_construct(p, args)
        output_patient = PatientPipeline(p, args).construct()
        output_studies = StudiesPipeline(output_series, args).construct()
        MergeSavePipeline(output_patient, output_studies, args).construct()
    
    return constructor



def run_prefetched_series(argv):
    """

    Args:
        argv:
    :return:
    """
    patient_series = SeriesFilter.batch_series_studies_by_patient()
    batch_size=15
    i=0
    lines = []

    for p in patient_series:
        if (len(lines) + len(p)) > batch_size:
            run_pipeline(argv, construct_sequence_pipeline(argv, {"SPECIFIC_GCS": lines}))
            client = storage.Client()
            bucket = client.get_bucket("ispy_dataquery")
            for b in os.listdir("output/"):
                name = f"output/small_{b}_{i}"
                print(name)
                blob = bucket.blob(name)
                blob.upload_from_filename(filename=f"output/{b}")
                os.remove(f"output/{b}")
            i += 1
            lines = []
        else:
            # Else merely add to series
            lines.extend(p)


def run_sequentially(argv):
    with open('SERIES.csv') as f:
        lines = [line.rstrip() for line in f]
    patient_series = SeriesFilter.batch_series_by_patient(lines)

    batch_size=60
    i=0
    lines = []

    for p in patient_series:
        if (len(lines) + len(p)) > batch_size:
            run_pipeline(argv, construct_sequence_pipeline(argv, {"SPECIFIC_SERIES": lines}))
            client = storage.Client()
            bucket = client.get_bucket("ispy_dataquery")
            for b in os.listdir("output/"):
                name = f"output/{i}_{b}"
                print(name)
                blob = bucket.blob(name)
                blob.upload_from_filename(filename=f"output/{b}")
                os.remove(f"output/{b}")
            i += 1
            lines = []
        else:
            # Else merely add to series
            lines.extend(p)

if __name__ == "__main__":
    run_prefetched_series(sys.argv)
       
