import numpy as np
from typing import Sequence
from pydantic import BaseModel

def sphere_intersection(ray_origin: np.ndarray, 
                        ray_direction: np.ndarray, 
                        sphere_center: np.ndarray, 
                        sphere_radius: float):
    
    a = np.linalg.norm(ray_direction)**2
    b = 2 * np.dot(ray_direction, ray_origin - sphere_center)
    c = np.linalg.norm(ray_origin - sphere_center)**2 - sphere_radius**2
    d = b**2 - 4 * a * c

    if d > 0:

        t1 = (-b + np.sqrt(d)) / (2 * a)
        t2 = (-b - np.sqrt(d)) / (2 * a)

        if t1 > 0 and t2 > 0:
            return min(t1, t2)
        
    return np.inf

class Sphere(BaseModel):
    radius: float
    center: Sequence

    ambient: Sequence
    diffuse: Sequence
    specular: Sequence
    shininess: float

    def intersection(self, ray_origin: np.ndarray, ray_direction: np.ndarray):
        return sphere_intersection(ray_origin, ray_direction, self.center, self.radius)

    @staticmethod
    def nearest_intersection_object(spheres: list['Sphere'], ray_origin: Sequence, ray_direction: Sequence):
        min_t = np.inf
        min_sphere_idx = None

        for i, sphere in enumerate(spheres):
            t = sphere.intersection(ray_origin, ray_direction)
            
            if t < min_t:
                min_t = t
                min_sphere_idx = i

        if min_sphere_idx is None or min_t == np.inf:
            return None, np.inf
        
        return spheres[min_sphere_idx], min_t
    
    @staticmethod
    def genenerate_random_spheres(n: int, radius_range: Sequence, center_range: Sequence, color_range: Sequence):
        spheres = []

        for _ in range(n):
            radius = np.random.uniform(*radius_range)
            center = np.random.uniform(*center_range, size=3).tolist()
            ambient = np.random.uniform(*color_range, size=3).tolist()
            diffuse = np.random.uniform(*color_range, size=3).tolist()
            specular = np.random.uniform(*color_range, size=3).tolist()
            shininess = np.random.uniform(0, 1)

            spheres.append(Sphere(radius=radius, center=center, ambient=ambient, diffuse=diffuse, specular=specular, shininess=shininess))

        return spheres