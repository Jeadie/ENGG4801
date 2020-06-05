class DICOMAccessError(Exception):
    """Exception raised when a DICOM could not be accessed or loaded into memory."""

    pass


class SeriesConstructionError(Exception):
    """Exception raised when a Series Object could not be constructed."""

    pass


class SeriesMetadataError(Exception):
    """Except raised when Series metadata could not be parsed."""

    pass
