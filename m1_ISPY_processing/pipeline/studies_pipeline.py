import argparse
import sys
from typing import Dict, Tuple

import apache_beam as beam
from apache_beam.pvalue import PCollection

import pipeline.constants as constants
from pipeline.custom_types import Types
import pipeline.util as util


def construct(series_collection: PCollection) -> PCollection:
    """ The Studies pipeline as documented.

    Returns:
         The final PCollection from the patient pipeline.
    """
    group_by_patients = (
        series_collection
        | "Parse out Study ID from each Series Obj"
        >> beam.Map(lambda x: (x[1].get("Study Instance UID"), x))
        | "Group by Study ID" >> beam.GroupByKey()
        | "Parse out Patient ID" >> beam.Map(parse_patient_from_study)
        | "Group by Patient ID" >> beam.GroupByKey()
    )
    return group_by_patients


def parse_patient_from_study(study: Types.StudyObj) -> Tuple[str, Types.StudyObj]:
    """ Turns an element in a PCollection into a keyed, by patient id, element.

    Indexing:
        series= study[1]
        singular_series = series[0]
        singular_series_metadata = singular_series[1]

    Args:
        study: A single Study Object.
    A keyed element Tuple of the form (Patient ID, Study Object).
    """
    return (study[1][0][1].pop("Clinical Trial Subject ID"), study)
