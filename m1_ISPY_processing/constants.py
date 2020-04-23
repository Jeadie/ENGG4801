from enum import Enum

import pydicom as dicom

import util

# TODO: Consider turning these into an Enum.
####################################
########### CLI Arguments ##########
####################################
PATIENT_OUTCOME_CSV_FILE_KEY = "patient_outcomes"
PATIENT_CLINICAL_CSV_FILE_KEY = "patient_clinical"
STUDIES_PATH="studies_dir"
SERIES_LIMIT="num-series"
TFRECORD_PATH="tfrecord_path"
NUM_TFRECORD_SHARDS="num_shards"

####################################
### MERGE+SAVE PIPELINE CONSTANTS ##
####################################
TFRECORD_SUFFIX=".tfrecords"


####################################
##### SERIES PIPELINE CONSTANTS ####
####################################
LOCAL_SERIES_FILTER_FILE = "ISPY1_MetaData.csv"
GCS_PREFIX="gs://"
BIGQUERY_STUDY_ID_HEADER = 'StudyInstanceUID'
BIGQUERY_SERIES_ID_HEADER = 'SeriesInstanceUID'
BIGQUERY_SERIES_QUERY= "SELECT DISTINCT StudyInstanceUID, SeriesInstanceUID FROM `chc-tcia.ispy1.ispy1` GROUP BY StudyInstanceUID, SeriesInstanceUID"


# NOTE: DICOM_SPECIFIC_TYPES is used instead of list(DICOM_TYPE_CONVERSION.keys()) for performance reasons.
DICOM_SPECIFIC_TYPES = [dicom.uid.UID, dicom.valuerep.DSfloat, dicom.valuerep.IS, dicom.valuerep.PersonName3, dicom.dataset.Dataset, dicom.multival.MultiValue]

def test_multivalue(x):
    if type(x[0]) in DICOM_SPECIFIC_TYPES:
        return [DICOM_TYPE_CONVERSION[type(x[0])](i) for i in x]
    else:
        return [x.type_constructor(i) for i in x]

DICOM_TYPE_CONVERSION = {
        dicom.valuerep.DSfloat: float,
        dicom.valuerep.IS: int,
        dicom.valuerep.PersonName3: str,
        dicom.dataset.Dataset: util.construct_metadata_from_DICOM_dictionary,
        dicom.multival.MultiValue: test_multivalue,
        dicom.uid.UID: str,
        bytes: lambda x: x.decode()
}

DICOM_PIXEL_TAG = (0x7fe0, 0x0010)

####################################
#### PATIENT PIPELINE CONSTANTS ####
####################################
CSV_DELIMETER = ","


class CSVHeader(Enum):
    SUBJECT_ID = "SUBJECTID"

    # Clinical
    AGE="age"
    RACE="race_id"
    ERpos='ERpos'
    PgRpos='PgRpos'
    HRPos='HR Pos'
    HER2STATUS= 'Her2MostPos'
    TRIPLE_LEVEL_HER = 'HR_HER2_CATEGORY'
    BILATERAL_CANCER='BilateralCa'
    LATERALITY='Laterality'
    LD_BASELINE='MRI LD Baseline'
    LD_POST_AC = 'MRI LD 1-3dAC'
    LD_INTER_REG = 'MRI LD InterReg'
    LD_PRE_SURGERY = 'MRI LD PreSurg'

    # Outcomes
    DATE="DataExtractDt"
    SURVIVAL_INDICATOR = "sstat"
    SURVIVAL_DURATION="survDtD2 (tx)"
    RECURRENCE_FREE_INDICATOR = "rfs_ind"
    RECURRENCE_FREE_DURATION="RFS"
    PATHOLOGICAL_COMPLETE_RESPONSE="PCR"
    RESIDUAL_CANCER_BURDEN_CLASS = "RCBClass"


CLINICAL_CSV_HEADERS = [
   CSVHeader.SUBJECT_ID.value,
   CSVHeader.DATE.value,
   CSVHeader.AGE.value,
   CSVHeader.RACE.value,
   CSVHeader.ERpos.value,
   CSVHeader.PgRpos.value,
   CSVHeader.HRPos.value,
    CSVHeader.HER2STATUS.value,
    CSVHeader.TRIPLE_LEVEL_HER.value,
    'HR_HER2_STATUS',
   CSVHeader.BILATERAL_CANCER.value,
   CSVHeader.LATERALITY.value,
   CSVHeader.LD_BASELINE.value,
   CSVHeader.LD_POST_AC.value,
   CSVHeader.LD_INTER_REG.value,
   CSVHeader.LD_PRE_SURGERY.value
]
OUTCOME_CSV_HEADERS = [
    CSVHeader.SUBJECT_ID.value,
    CSVHeader.DATE.value,
    CSVHeader.SURVIVAL_INDICATOR.value,
    CSVHeader.SURVIVAL_DURATION.value,
    CSVHeader.RECURRENCE_FREE_DURATION.value,
    CSVHeader.RECURRENCE_FREE_INDICATOR.value,
    CSVHeader.PATHOLOGICAL_COMPLETE_RESPONSE.value,
    CSVHeader.RESIDUAL_CANCER_BURDEN_CLASS.value,
]

# SUBJECTID is not duplicated in joint CSV rows.
JOINT_CSV_HEADERS = OUTCOME_CSV_HEADERS + CLINICAL_CSV_HEADERS[1:]
