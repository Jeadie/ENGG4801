import argparse
import os
import sys
from typing import Dict, List
import logging

import apache_beam as beam
from google.cloud import storage

from pipeline.merge_and_save_pipeline import construct as MergeSavePipeline
from pipeline.patient_pipeline import construct as PatientPipeline
from pipeline.series_pipeline_gcs import construct as SeriesPipeline
from pipeline.series_filter import SeriesFilter
from pipeline.studies_pipeline import construct as StudiesPipeline
from util import run_pipeline


def main(argv: List[str]) -> int:
    """Main program run for data processing."""
    return run_pipeline(argv, construct_main_pipeline)


def main_pipeline(args: Dict[str, object], p: beam.Pipeline):
    output_series = SeriesPipeline(p, args)
    output_patient = PatientPipeline(p, args)
    output_studies = StudiesPipeline(output_series)
    MergeSavePipeline(output_patient, output_studies, args)


def construct_main_pipeline(parsed_args: argparse.Namespace, p: beam.Pipeline) -> None:
    """ Responsible for constructing the main pipeline for processing ISPY1.

    Args:
        parsed_args: CLI arguments parsed and validated.
    :param p: A pipeline, preconfigured with pipeline options.
    """
    return main(vars(parsed_args), p)


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
        main_pipeline(args, p)

    return constructor


def run_prefetched_series(argv):
    """

    Args:
        argv:
    :return:
    """
    patient_series = SeriesFilter.batch_series_studies_by_patient()
    batch_size = 15
    i = 0
    lines = []

    for p in patient_series:
        if (len(lines) + len(p)) > batch_size:
            run_pipeline(
                argv, construct_sequence_pipeline(argv, {"SPECIFIC_GCS": lines})
            )
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

def dataflow(argv):
    patient_series = SeriesFilter.batch_series_studies_by_patient()
    flat_series = [item for sublist in patient_series for item in sublist]
    run_pipeline(
     argv, construct_sequence_pipeline(argv, {"SPECIFIC_GCS": flat_series})
    )
    print("DONE")

def run_sequentially(argv):
    with open("SERIES.csv") as f:
        lines = [line.rstrip() for line in f]
    patient_series = SeriesFilter.batch_series_by_patient(lines)

    batch_size = 60
    i = 0
    lines = []

    for p in patient_series:
        if (len(lines) + len(p)) > batch_size:
            run_pipeline(
                argv, construct_sequence_pipeline(argv, {"SPECIFIC_SERIES": lines})
            )
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
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)

    dataflow(sys.argv)
