import pandas as pd

import constants


class SeriesFilter(object):
    """ Object responsible for filtering series based on user requirements."""
    USED_MRI = ['fl3d_sag_uni_R', 'ACRIN_6657/FL3D_T1_SAG_CA_', 'fl3d_sag_uni_L', 'Dynamic-Unilateral 3dfgre',
                'Dynamic-3dfgre', 'LT IR-SPGR SAG', 'ACRIN_6657/FL3D_T1_SAG_CA', 'BILAT T1 SPGR POST',
                'Dynamic-3dfgre-Phase=160', 'T1 right breast post', 'T1 left breast post', 'RT IR-SPGR SAG',
                'Penn-Lesion1/2-3DFatsat-Sa', 'T1 right breast', 'T1 left breast', '#6657 BreasFS3DSAG 6DYN',
                'ACRIN_6657/FL3D_T1_SAG_POS', '3D', 'BILAT T1 SPGR PRE', 'LEFT_ACRIN_6657/LEFT_FL3D_', 'IR-SPGR-SAG',
                '#6657 Breas3DSAG6D 1.37', 'DCE POST Right', 'DCE POST Left', 'T1 right breast post delay',
                'CTLMID,Sag,3D,SPGR,VBw, GRx,', 'LT IR-SPGR SAG PRE/POST', 'Penn-HighRisk/T1_3D_SE_SAG',
                'Penn-HighRisk/T1-3D-SAG-FS', 'Sag IR SPGR', 'T1 left breast post delay', 'T1 Sagittal post',
                't1_fl3d_uni_sag_fatsat', 'T1 Sagittal pre', 'ACRIN_6657/FL3D_T1_SAG', 'DCE PRE Left',
                't1_fl3d_sag_fatsat(new)', 'DCE PRE Right', 'BRSTCA SENSFS3DSAG 6DYN', 'NCI/HIGH_RES_SAG',
                'ACRIN_6657/LEFTFL3D_T1_SAG']
    USED_SEG = ('Breast Tissue Segmentation', "VOI Breast Tissue Segmentation")

    def __init__(self, filter_file="ISPY1_MetaData.csv"):
        self.df = pd.read_csv(filter_file, delimiter=",", header=0)[["Series UID", "Series Description"]]

    def filter_series_path(self, path: str) -> bool:
        """ Returns True if this Series path should be included in the pipeline, False otherwise.

        Args:
            path: GCS path.
        """
        series_id = path.split("/")[-2]
        series_description = self.df[self.df["Series UID"] == series_id]["Series Description"]
        if series_description.count() == 0:
            return False
        print("series_description", list(series_description)[0])
        return list(series_description)[0] in SeriesFilter.USED_SEG or list(series_description)[0] in SeriesFilter.USED_MRI
