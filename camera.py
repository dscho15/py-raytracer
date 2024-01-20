import numpy as np
from rotations import rpy
from pydantic import BaseModel
from typing import Sequence

def intrinsics(f, cx, cy):
    return np.array([[f, 0, cx],
                     [0, f, cy],
                     [0, 0, 1]])

def extrinsic(rpy_, t_):
    I = np.eye(4, 4)
    I[:3, :3] = rpy(*rpy_)
    I[:3, 3] = t_
    return I

class Camera(BaseModel):
    focal_length: float = 256
    width: int = 256
    height: int = 256

    t: Sequence = np.r_[0, 0, -5]
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    
    @property
    def cx(self):
        return self.width // 2
    
    @property
    def cy(self):
        return self.height // 2
    
    @property
    def h(self):
        return self.height
    
    @property
    def w(self):
        return self.width
    
    @property
    def intrinsics(self):
        return intrinsics(self.focal_length, self.cx, self.cy)
    
    @property
    def extrinsics(self):
        return extrinsic((self.roll, self.pitch, self.yaw), self.t)
    
    @property
    def proj_matrix(self):
        return self.intrinsics @ np.eye(3, 4) @ self.extrinsics
    
    @property
    def ray_matrix(self):
        return np.eye(3, 4) @ self.extrinsics
    
    @property
    def origin(self):
        return self.t


camera = Camera()
print(camera)