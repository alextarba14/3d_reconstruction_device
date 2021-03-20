import pyrealsense2 as rs
import numpy as np
import time


def get_difference_item(array, current_data, index):
    """Computes differences between current data and the previous data from array"""
    if index <= 0:
        return [current_data.x, current_data.y, current_data.z]
    dx = current_data.x - array[index - 1][0]
    dy = current_data.y - array[index - 1][1]
    dz = current_data.z - array[index - 1][2]

    return [dx, dy, dz]


def get_vertices_and_texture_from_pointcloud(points: rs.points, color_frame: rs.frame):
    # Reference: https://github.com/Resays/xyz_rgb_realsense/blob/ede2bf9cc81d67ff0a7b616a5c70ff529e43bfe3/xyz_rgb_realsense.cpp
    start_time = time.time()
    vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)  # xyz
    tex_coords = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)  # uv
    # get information from color frame
    texture_data = np.asanyarray(color_frame.get_data()).view(np.uint8).reshape(-1, 1)
    w, h = color_frame.get_width(), color_frame.get_height()
    bytes_per_pixel = color_frame.get_bytes_per_pixel()
    stride_in_bytes = color_frame.get_stride_in_bytes()

    updated_vertices = []
    texture = []
    length = points.size()
    for i in range(length):
        # check if depth is greater than zero to exclude invalid points
        if vertices[i][2] > 0:
            updated_vertices.append(vertices[i])
            texture.append(get_texture_color(tex_coords[i], texture_data, w, h, bytes_per_pixel, stride_in_bytes))

    updated_vertices = np.asanyarray(updated_vertices).reshape(-1,3)
    texture = np.asanyarray(texture).reshape(-1,3)
    print("It took: ", time.time()-start_time)


def get_texture_color(tex_coords: rs.texture_coordinate, texture_data, w, h, bytes_per_pixel, stride_in_bytes):
    """
    Returns a tuple in (R,G,B) format.
    """
    x = min(max(int(tex_coords[0] * w + 0.5), 0), w - 1)
    y = min(max(int(tex_coords[1] * h + 0.5), 0), h - 1)
    idx = x * bytes_per_pixel  + y * stride_in_bytes
    return [texture_data[idx], texture_data[idx + 1], texture_data[idx + 2]]
