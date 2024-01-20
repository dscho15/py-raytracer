from rotations import rpy
from triangle import parameterize_triangle_as_plane
import numpy as np

def check_point_inside_triangle(point: np.ndarray,
                                triangle_vertices: np.ndarray):
    
    # get the vectors of the triangle
    v1 = triangle_vertices[1] - triangle_vertices[0]
    v2 = triangle_vertices[2] - triangle_vertices[0]
    v3 = point - triangle_vertices[0]

    # calculate the dot products
    dot_product_1 = np.dot(np.cross(v1, v2), np.cross(v1, v3))
    dot_product_2 = np.dot(np.cross(v2, v1), np.cross(v2, v3))
    dot_product_3 = np.dot(np.cross(v3, v1), np.cross(v3, v2))

    # check if the point is inside the triangle
    if dot_product_1 >= 0 and dot_product_2 >= 0 and dot_product_3 >= 0:
        return True
    else:
        return False

def check_ray_plane_intersection(ray_origin: np.ndarray, 
                                 ray_direction: np.ndarray, 
                                 plane_equation: np.ndarray):
    # get the normal vector of the plane
    a, b, c, d = plane_equation
    normal_vector = np.array([a, b, c])

    # get the intersection point of the ray and the plane
    t = -(d + np.dot(normal_vector, ray_origin)) / np.dot(normal_vector, ray_direction)
    intersection_point = ray_origin + t * ray_direction

    return intersection_point

def check_ray_triangle_intersection(ray_origin: np.ndarray, 
                                    ray_direction: np.ndarray, 
                                    triangle_vertices: np.ndarray):
    # get the plane equation
    plane_equation = parameterize_triangle_as_plane(triangle_vertices)
    print("plane equation:\n", plane_equation)

    # get the intersection point of the ray and the plane
    intersection_point = check_ray_plane_intersection(ray_origin, ray_direction, plane_equation)
    print("intersection point:\n", intersection_point)

    # check if the intersection point is inside the triangle
    if check_point_inside_triangle(intersection_point, triangle_vertices):
        return True
    else:
        return False