from pydantic import BaseModel
import numpy as np
from typing import Sequence

class Light(BaseModel):
    position: Sequence = np.r_[0, 0, 0]
    ambient: Sequence = np.r_[0.5, 0.5, 0.5]
    diffuse: Sequence = np.r_[0.5, 0.5, 0.5]
    specular: Sequence = np.r_[0.5, 0.5, 0.5]
    