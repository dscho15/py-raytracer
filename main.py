import numpy as np

from rotations import rpy
from geometric_primitives.sphere import Sphere
from camera import Camera
from pprint import pprint
from rotations import quaternion_to_rotation_matrix, rpy, slerp
from itertools import product
from functools import partial
import cProfile
import random

from time import time

# decimals in print
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

# define a bunch of spheres
spheres = Sphere.genenerate_random_spheres(2, (0.5, 1.0), (-0.5, 0.5), (0.0, 1.0))

# define a bunch of lights
light = {"position": np.array([0, 0, -3]), "ambient": np.array([0.5, 0.5, 0.5])}
camera = Camera()

inv_intrinsics = np.linalg.inv(camera.intrinsics)

imgs = []
q0 = np.r_[(0, 0, 1, 0)]
q1 = np.r_[(0, 0, 0, 1)]

q = slerp(q0, q1, 0.5)

R = quaternion_to_rotation_matrix(q)

# render image
img = np.zeros((camera.h, camera.w, 3))

time0 = time()

def normalize(v):
    return v / np.linalg.norm(v)

def color_pixel(i, j, camera, spheres):
    ray_dir = camera.rays()[i, j]
    
    (sphere, dist) = Sphere.nearest_intersection_object(spheres, camera.origin, ray_dir)

    if dist == np.inf:
        return np.array([0., 0., 0.])

    intersection = camera.origin + dist * ray_dir
    surface_normal = normalize(intersection - sphere.center)
    shifted_point = intersection + 1e-5 * surface_normal
    intersection_to_light = normalize(light["position"] - shifted_point)

    _, min_dist = Sphere.nearest_intersection_object(spheres, shifted_point, intersection_to_light)
    intersection_to_light_distance = np.linalg.norm(light['position'] - intersection)
    is_shadowed = min_dist < intersection_to_light_distance

    if is_shadowed:
        return np.array([0., 0., 0.])
    
    illumination = np.r_[0., 0., 0.]
    illumination += sphere.ambient * light["ambient"]

    return sphere.ambient

p_color_pixel = partial(color_pixel, camera=camera, spheres=spheres)
for i, j in product(range(camera.h), range(camera.w)):
    img[i, j] = p_color_pixel(i, j)

time1 = time()
print("time elapsed: ", time1 - time0)

# save image
from PIL import Image
img = Image.fromarray((img * 255).astype(np.uint8))
img.save("ray_sphere.png")