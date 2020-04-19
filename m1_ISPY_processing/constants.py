from enum import Enum

####################################
########### CLI Arguments ##########
####################################
PATIENT_OUTCOME_CSV_FILE_KEY = "patient_outcomes"
PATIENT_CLINICAL_CSV_FILE_KEY = "patient_clinical"
STUDIES_PATH="studies_dir"
SERIES_LIMIT="num-series"

####################################
##### SERIES PIPELINE CONSTANTS ####
####################################
GCS_PREFIX="gs://"
BIGQUERY_STUDY_ID_HEADER = 'StudyInstanceUID'
BIGQUERY_SERIES_ID_HEADER = 'SeriesInstanceUID'
BIGQUERY_SERIES_QUERY= "SELECT DISTINCT StudyInstanceUID, SeriesInstanceUID FROM `chc-tcia.ispy1.ispy1` GROUP BY StudyInstanceUID, SeriesInstanceUID"

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
