import numpy as np

from rotations import rpy
from geometric_primitives.sphere import Sphere
from camera import Camera
from pprint import pprint
from rotations import quaternion_to_rotation_matrix, rpy, slerp
import random

from time import time

# decimals in print
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

# define a bunch of spheres
spheres = []
for _ in range(5):  # create 5 random spheres
    
    center = np.random.rand(3) * 2  # random center coordinates in the range [0, 10)
    radius = random.uniform(0.5, 1.5)  # random radius in the range [0.5, 1.5)
    ambient = np.random.rand(3)  # random ambient color
    
    diffuse = np.random.rand(3)  # random diffuse color
    specular = np.random.rand(3)  # random specular color
    shininess = random.uniform(0, 1)  # random shininess in the range [0, 1)

    sphere_data = {
        "radius": radius,
        "center": center.tolist(),
        "ambient": ambient.tolist(),
        "diffuse": diffuse.tolist(),
        "specular": specular.tolist(),
        "shininess": shininess
    }

    spheres.append(Sphere(**sphere_data))

# define a bunch of lights
lights = []
lights.append((np.array([1, 1, 0]), 1.0))
camera = Camera()

inv_intrinsics = np.linalg.inv(camera.intrinsics)

imgs = []

# move the camera 
# linear interpolate based on a quaterion 

q0 = np.r_[(0, 0, 1, 0)]
q1 = np.r_[(0, 0, 0, 1)]

# 
q = slerp(q0, q1, 0.5)

R = quaternion_to_rotation_matrix(q)

# render image
img = np.zeros((camera.h, camera.w, 3))

time0 = time()

for i, y in enumerate(range(camera.h)):

    for j, x in enumerate(range(camera.w)):

        pix = np.array([x, y, 1])
        ray_dir = inv_intrinsics @ pix
        ray_dir = ray_dir / np.linalg.norm(ray_dir)
        
        (sphere, dist) = Sphere.nearest_intersection_object(spheres, camera.origin, ray_dir)

        if dist == np.inf:
            continue

        img[i, j] = sphere.ambient

time1 = time()
print("time elapsed: ", time1 - time0)

# save image
from PIL import Image
img = Image.fromarray((img * 255).astype(np.uint8))
img.save("ray_sphere.png")