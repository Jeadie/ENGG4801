from typing import Dict, List, Tuple

import apache_beam as beam
from apache_beam.pvalue import PCollection
import tensorflow as tf

import constants
from custom_types import Types
from util_tensorflow import int64_feature, float_feature, int64List_feature


class MergeSavePipeline(object):
    """Pipeline responsible for merging studies with Patient data and saving each to TFRecords."""

    def __init__(
        self,
        output_patient: PCollection,
        output_studies: PCollection,
        argv: Dict[str, object],
    ) -> None:
        """ Constructor.

        Args:
            output_patient: The outputted PCollection from the Patient Pipeline.
            output_studies: The outputted PCollection from the Studies Pipeline.
            argv: Parsed CLI arguments.
        """
        self.patient = output_patient
        self.studies = output_studies
        self.settings = argv

    def construct(self) -> None:
        """ The Merge and Save pipeline as documented.

        Returns:
             None. No result is returned. This pipeline writes to all final data sinks.
        """
        # Map PCollections to Keyed PCollections.
        patient_key = self.patient | "Create Keyed PCollection for Patient" >> beam.Map(
            lambda x: (str(x["patient_id"]), x)
        )
        tf_examples = (
            {"studies": self.studies, "patient": patient_key}
            | "Join Patient and Studies by Patient Key" >> beam.CoGroupByKey()
            | "Filter out Patients without Images or vice versa"
            >> beam.Filter(lambda x: (x[1]["studies"] != []))
            | "Convert to tf.Examples"
            >> beam.Map(MergeSavePipeline.convert_to_tf_example)
        )

        (
            tf_examples
            | "Serialise tf.Example" >> beam.Map(lambda x: x.SerializeToString())
            | "Save to TFRecord"
            >> beam.io.WriteToTFRecord(
                file_path_prefix=self.settings[constants.TFRECORD_NAME],
                file_name_suffix=constants.TFRECORD_SUFFIX,
                num_shards=self.settings[constants.NUM_TFRECORD_SHARDS],
            )
        )

    @staticmethod
    def convert_to_tf_example(
        patient_data: Tuple[str, Dict[str, object]]
    ) -> tf.train.Example:
        """ Converts an element from the combined patient+study PCollection into a TF Example.

        Args:
            patient_data: A single patient's data with clinical, outcome and imaging data.

        Returns:
            An Example ready to be serialised and saved out of memory (as a TFRecord generally)
        """
        data = patient_data[1]
        patient = data["patient"][0]
        studies = data["studies"][0]
        features = convert_patient_to_feature(patient)
        for study_id, study in studies:
            study_data = convert_study_to_feature(study)
            for feature in study_data:
                features.update(feature)

        return tf.train.Example(
            features=tf.train.Features(feature=features),
        )


def convert_series_to_feature(series: Types.SeriesObj,) -> Dict[str, tf.train.Feature]:
    """ Converts a single SeriesObj to a feature dictionary.

    Args:
        series: A series object. Expected keys in Metadata are:
            "Pixel Spacing", "Spacing Between Slices", "Modality", "Laterality",
            and either "time" and "flag"
                or "Series Instance UID" and "Study Instance UID"

    Returns:
        A feature dictionary for a single Series.
    """
    image, metadata = series
    if metadata.get("flags") and metadata.get("time"):
        name = f"time{metadata.get('time')[1:]}/{'_'.join(metadata.get('flags'))}/"
    else:
        name = f"{metadata.get('Study Instance UID')}/{metadata.get('Series Instance UID')}/"
    return dict([(f"{name}{k}", v) for (k, v) in {
            "image": int64List_feature(image.flatten().tolist()),
            "dx": float_feature(metadata.get("Pixel Spacing")[0]),
            "dy": float_feature(metadata.get("Pixel Spacing")[1]),
            "dz": float_feature(metadata.get("Spacing Between Slices")),
            "is_seg": int64_feature(int(metadata.get("Modality") == "SEG")),
            "right": int64_feature(int(metadata.get("Laterality") == "R")),
            "shape": int64List_feature(image.shape),
        }.items()])


def convert_study_to_feature(study: List[Types.SeriesObj]) -> List[Dict[str, tf.train.Feature]]:
    """ Convert a single Study (parsed differently to a Types.StudyObj) into a list of
        feature dictionaries, each corresponding to a single Series.

    Args:
        Study: A list of Series objects.

    Returns:
        A list of Series feature dictionaries.
    """
    return [convert_series_to_feature(s) for s in study]


def convert_patient_to_feature(
    patient_data: Dict[str, object]
) -> Dict[str, tf.train.Feature]:
    """ Converts a patient's metadata to a Tensorflow feature dictionary.

    Args:
        patient_data: Relevant patient metadata.

    Returns:
        Features to include specific to the patient.
    """
    # TODO: Maybe prefix with "patient/" for post processing ease.
    return {
        "patient_id": int64_feature(patient_data.get("patient_id")),
        "age": float_feature(patient_data.get("demographic_metadata").get("age")),
        "race": int64_feature(patient_data.get("demographic_metadata").get("race")),
        "ERpos": int64_feature(patient_data.get("clinical").get("ERpos")),
        "Pgpos": int64_feature(patient_data.get("clinical").get("Pgpos")),
        "HRpos": int64_feature(patient_data.get("clinical").get("HRpos")),
        "HER_two_status": int64_feature(
            patient_data.get("clinical").get("HER_two_status")
        ),
        "three_level_HER": int64_feature(
            patient_data.get("clinical").get("three_level_HER")
        ),
        "Bilateral": int64_feature(patient_data.get("clinical").get("Bilateral")),
        "Laterality": int64_feature(patient_data.get("clinical").get("Laterality")),
        # Outcomes
        "Sstat": int64_feature(patient_data.get("outcome").get("Sstat")),
        "survival_duration": int64_feature(
            patient_data.get("outcome").get("survival_duration")
        ),
        "rfs_ind": int64_feature(patient_data.get("outcome").get("rfs_ind")),
        "rfs_duration": int64_feature(patient_data.get("outcome").get("rfs_duration")),
        "pCR": int64_feature(patient_data.get("outcome").get("pCR")),
        "RCB": int64_feature(patient_data.get("outcome").get("RCB")),
        "LD": int64List_feature(patient_data.get("LD"))

    }
