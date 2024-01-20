import numpy as np

def triangle(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, c1: np.ndarray, c2: np.ndarray, c3: np.ndarray):

    # create a triangel with three vertices and color for each vertex
    vertices = np.array([x1, x2, x3])
    colors = np.array([c1, c2, c3])

    return vertices, colors

def parameterize_triangle_as_plane(triangle_vertices: np.ndarray):

    # get the normal vector of the plane
    p1, p2, p3 = triangle_vertices
    v1 = p3 - p1
    v2 = p2 - p1
    normal_vector = np.cross(v1, v2)

    # get the plane equation
    a, b, c = normal_vector
    d = -np.dot(normal_vector, p3)
    plane_equation = np.array([a, b, c, d])

    return plane_equation