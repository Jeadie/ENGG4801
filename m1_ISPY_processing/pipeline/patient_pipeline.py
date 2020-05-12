from typing import Dict, List

import apache_beam as beam
from apache_beam.pvalue import PCollection
from apache_beam.pipeline import Pipeline

from . import constants
from .constants import CSVHeader #import CSVHeader


def construct(pipeline: Pipeline, argv: Dict[str, object]) -> PCollection:
    """ The patient Pipeline as documented.

    Args:
        pipeline: A reference to the main pipeline.
        argv: Parsed arguments from CLI.

    Returns:
         The final PCollection from the patient pipeline.
    """
    print("PATIENTS")
    patients = get_all_patients(pipeline, argv)
    return patients | "Parse + Format patient metadata" >> beam.Map(
        format_patient_metadata
    )


def format_patient_metadata(patient: List[str]) -> Dict[str, object]:
    """ Parses a single CSV line and extracts and formats the clinical and outcome data.

    Args:
        patient: A single line from the joint CSV of patient clinical and outcome data.

    Return:

    """
    data = dict(
        filter(lambda x: x[-1] != "", zip(constants.JOINT_CSV_HEADERS, patient))
    )
    return {
        "patient_id": int(data[CSVHeader.SUBJECT_ID.value]),
        "demographic_metadata": {
            "age": float(data.get(CSVHeader.AGE.value, -1)),
            "race": int(data.get(CSVHeader.RACE.value, -1)),
        },
        "clinical": {
            "ERpos": int(data.get(CSVHeader.ERpos.value, -1)),
            "Pgpos": int(data.get(CSVHeader.PgRpos.value, -1)),
            "HRpos": int(data.get(CSVHeader.HRPos.value, -1)),
            "HER_two_status": int(data.get(CSVHeader.HER2STATUS.value, -1)),
            "three_level_HER": int(data.get(CSVHeader.TRIPLE_LEVEL_HER.value, -1)),
            "Bilateral": int(data.get(CSVHeader.BILATERAL_CANCER.value, -1)),
            "Laterality": int(data.get(CSVHeader.LATERALITY.value, -1)),
        },
        "LD": [
            int(data.get(x.value, -1))
            for x in [
                CSVHeader.LD_BASELINE,
                CSVHeader.LD_POST_AC,
                CSVHeader.LD_INTER_REG,
                CSVHeader.LD_PRE_SURGERY,
            ]
        ],
        "outcome": {
            "Sstat": int(data.get(CSVHeader.SURVIVAL_INDICATOR.value)),
            "survival_duration": int(data.get(CSVHeader.SURVIVAL_DURATION.value)),
            "rfs_ind": int(data.get(CSVHeader.RECURRENCE_FREE_INDICATOR.value)),
            "rfs_duration": int(data.get(CSVHeader.RECURRENCE_FREE_DURATION.value)),
            "pCR": int(data.get(CSVHeader.PATHOLOGICAL_COMPLETE_RESPONSE.value, -1)),
            "RCB": int(data.get(CSVHeader.RESIDUAL_CANCER_BURDEN_CLASS.value, -1)),
        },
    }


def get_all_patients(pipeline: Pipeline, settings: Dict[str, object]) -> PCollection:
    """ Parses the two CSV files, outcomes and clinical, and merges them based on PatientID.

    It is assumed the two CSVs are ordered, ascendingly, by PatientID.

    Args:
        pipeline: A reference to the main pipeline.
        settings: Parsed arguments from CLI.

    Returns:
        A PCollection of unparsed CSV data merged from the two CSV pages.
    """

    # Both of these create PCollections with iterators that __next__() -> List[str]

    outcomes_data = load_csv(
        pipeline, settings[constants.PATIENT_OUTCOME_CSV_FILE_KEY]
    ) | "Parse Outcomes' Patient ID" >> beam.Map(lambda x: (x[0], x[1:]))
    clinical_data = load_csv(
        pipeline, settings[constants.PATIENT_CLINICAL_CSV_FILE_KEY]
    ) | "Parse Clinical' Patient ID" >> beam.Map(lambda x: (x[0], x[1:]))
    return (
        {"outcomes": outcomes_data, "clinical": clinical_data}
        | "Combine outcomes & clinical" >> beam.CoGroupByKey()
        | "Flatten outcomes & clinical" >> beam.Map(flatten_patient_data)
    )


def load_csv(
    pipeline: Pipeline, csv_path: str, split=constants.CSV_DELIMETER
) -> PCollection:
    """ Loads a CSV from a file.

    Args:
        pipeline: A reference to the main pipeline (to construct within).
        csv_path: Path to the CSV to load.

    Returns:
        A PCollection whereby each element is a row from the CSV with type, List[str].
    """
    return (
        pipeline
        | f"Read CSV: {csv_path}" >> beam.io.ReadFromText(csv_path, skip_header_lines=1)
        | f"Split CSV: {csv_path}" >> beam.Map(lambda x: x.split(split))
    )


def flatten_patient_data(patient):
    """ Flattens a CoGroupByKey.

    :param patient:
    :return:
    """
    patient_id = patient[0]
    outcomes = patient[1]["outcomes"][0]
    clinical = patient[1]["clinical"][0]

    return [patient_id] + outcomes + clinical
