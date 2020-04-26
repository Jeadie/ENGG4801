from typing import Dict, List, Tuple

import numpy as np


class Types(object):
    SeriesObj = Tuple[np.array, Dict[str, object]]
    StudyObj = Tuple[str, List[SeriesObj]]
