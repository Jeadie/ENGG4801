from typing import List, Tuple

import pandas as pd


class SeriesFilter(object):
    """ Object responsible for filtering series based on user requirements."""

    USED_MRI = {
        "fl3d_sag_uni_R": ["MRI", "FL3D", "SAG", "RIGHT"],
        "ACRIN_6657/FL3D_T1_SAG_CA_": ["MRI", "FL3D", "T1", "SAG"],
        "fl3d_sag_uni_L": ["MRI", "FL3D", "SAG", "LEFT"],
        "Dynamic-Unilateral 3dfgre": ["MRI", "DCE"],
        "Dynamic-3dfgre": ["MRI", "DCE"],
        "LT IR-SPGR SAG": ["MRI", "SAG", "LEFT", "T1"],
        "ACRIN_6657/FL3D_T1_SAG_CA": ["MRI", "FL3D", "T1", "SAG"],
        "BILAT T1 SPGR POST": ["MRI", "T1"],
        "Dynamic-3dfgre-Phase=160": ["MRI", "DCE"],
        "T1 right breast post": ["MRI", "T1", "RIGHT", "POST"],
        "T1 left breast post": ["MRI", "T1", "LEFT", "POST"],
        "RT IR-SPGR SAG": ["MRI", "RIGHT", "SAG"],
        "Penn-Lesion1/2-3DFatsat-Sa": ["MRI", "SAG"],
        "T1 right breast": ["MRI", "T1", "RIGHT"],
        "T1 left breast": ["MRI", "T1", "LEFT"],
        "#6657 BreasFS3DSAG 6DYN": ["MRI", "SAG"],
        "ACRIN_6657/FL3D_T1_SAG_POS": ["MRI", "FL3D", "T1", "SAG"],
        "3D": ["MRI"],
        "BILAT T1 SPGR PRE": ["MRI", "T1"],
        "LEFT_ACRIN_6657/LEFT_FL3D_": ["MRI", "LEFT", "FLASH"],
        "IR-SPGR-SAG": ["MRI", "SAG", "T1"],
        "#6657 Breas3DSAG6D 1.37": ["MRI", "SAG"],
        "DCE POST Right": ["MRI", "DCE", "RIGHT"],
        "DCE POST Left": ["MRI", "DCE", "LEFT"],
        "T1 right breast post delay": ["MRI", "T1", "RIGHT", "POST"],
        "CTLMID,Sag,3D,SPGR,VBw, GRx,": ["MRI", "SAG"],
        "LT IR-SPGR SAG PRE/POST": ["MRI", "SAG"],
        "Penn-HighRisk/T1_3D_SE_SAG": ["MRI", "T1", "SAG"],
        "Penn-HighRisk/T1-3D-SAG-FS": ["MRI", "T1", "SAG"],
        "Sag IR SPGR": ["MRI", "SAG"],
        "T1 left breast post delay": ["MRI", "T1", "LEFT", "POST"],
        "T1 Sagittal post": ["MRI", "T1", "SAG"],
        "t1_fl3d_uni_sag_fatsat": ["MRI", "T1", "FL3D", "SAG"],
        "T1 Sagittal pre": ["MRI", "T1", "SAG"],
        "ACRIN_6657/FL3D_T1_SAG": ["MRI", "FL3D", "T1", "SAG"],
        "DCE PRE Left": ["MRI", "DCE", "LEFT"],
        "t1_fl3d_sag_fatsat(new)": ["MRI", "FL3D", "T1", "SAG"],
        "DCE PRE Right": ["MRI", "DCE", "RIGHT"],
        "BRSTCA SENSFS3DSAG 6DYN": ["MRI", "SAG"],
        "NCI/HIGH_RES_SAG": ["MRI", "SAG"],
        "ACRIN_6657/LEFTFL3D_T1_SAG": ["MRI", "FL3D", "T1", "SAG", "LEFT"],
        "5": ["MRI", "FL3D", "T1", "SAG", "LEFT"],
    }
    USED_SEG = {
        "Breast Tissue Segmentation": ["Tissue", "SEG"],
        "VOI Breast Tissue Segmentation": ["SEG", "VOI"],
    }

    def __init__(self, filter_file="ISPY1_MetaData.csv"):
        self.df = pd.read_csv(filter_file, delimiter=",", header=0)[
            ["Series UID", "Series Description"]
        ]

    @classmethod
    def batch_series_studies_by_patient(
        cls,
        descriptions="series_description.csv",
        series_study_file="series_studies.csv",
        filter_file="ISPY1_MetaData.csv",
    ) -> List[List[str]]:
        """

        :return:
        """
        df = pd.read_csv(filter_file, delimiter=",", header=0)[
            ["Series UID", "Patient Id"]
        ]
        data = pd.read_csv(series_study_file, delimiter=",", header=0)
        descriptions = pd.read_csv(descriptions, delimiter=",", header=0)

        df = df.rename(columns={"Series UID": "SeriesInstanceUID"})
        result = pd.merge(df, data, on=["SeriesInstanceUID", "SeriesInstanceUID"])
        result = result[["Patient Id", "StudyInstanceUID", "SeriesInstanceUID"]]
        descriptions = descriptions[
            descriptions["SeriesDescription"].isin(
                list(SeriesFilter.USED_SEG.keys()) + list(SeriesFilter.USED_MRI.keys())
            )
        ]

        result = result[
            result.SeriesInstanceUID.isin(descriptions["SeriesInstanceUID"])
        ]
        result["gs_path"] = result.apply(
            lambda x: f"gs://ispy_dataquery/dicoms/{x['StudyInstanceUID']}/{x['SeriesInstanceUID']}/",
            axis=1,
        )
        result = dict(result.groupby("Patient Id")["gs_path"].apply(list))
        return list(result.values())

    @classmethod
    def batch_series_by_patient(
        cls, lines, filter_file="ISPY1_MetaData.csv"
    ) -> List[List[str]]:
        """

        :return:
        """
        df = pd.read_csv(filter_file, delimiter=",", header=0)[
            ["Series UID", "Patient Id"]
        ]
        patients = list(set(list(df["Patient Id"])))
        return [list(df[df["Patient Id"] == p]["Series UID"]) for p in patients]

    def filter_series_path(self, path: str) -> bool:
        """ Returns True if this Series path should be included in the pipeline, False otherwise.

        Args:
            path: GCS path.
        """
        series_id = path.split("/")[-2]
        series_description = self.df[self.df["Series UID"] == series_id][
            "Series Description"
        ]
        if series_description.count() == 0:
            return False

        print("series_description", list(series_description))
        return (
            list(series_description)[0] in SeriesFilter.USED_SEG.keys()
            or list(series_description)[0] in SeriesFilter.USED_MRI.keys()
        )

    def get_series_flags(self, description: str) -> List[str]:
        """ Gets appropriate set of flags for a type of series.

        Args:
            description: The description of the Series.

        Return:
            A list of flags relevant to the Series.
        """
        return SeriesFilter.USED_SEG.get(
            description, SeriesFilter.USED_MRI.get(description, "")
        )
