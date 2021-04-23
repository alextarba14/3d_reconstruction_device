import time
import pyrealsense2 as rs
import numpy as np


def get_texture_from_pointcloud(vertices, tex_coords, color_frame: rs.frame):
    # Reference: https://github.com/Resays/xyz_rgb_realsense/blob/ede2bf9cc81d67ff0a7b616a5c70ff529e43bfe3/xyz_rgb_realsense.cpp
    start_time = time.time()
    # get information from color frame
    texture_data = np.asanyarray(color_frame.get_data()).view(np.uint8).reshape(-1, 1)
    w, h = color_frame.get_width(), color_frame.get_height()
    bytes_per_pixel = color_frame.get_bytes_per_pixel()
    stride_in_bytes = color_frame.get_stride_in_bytes()

    texture = []
    length = len(vertices)
    for i in range(length):
        texture.append(get_texture_color(tex_coords[i], texture_data, w, h, bytes_per_pixel, stride_in_bytes))

    print("It took: ", time.time() - start_time)
    return np.asanyarray(texture).reshape(-1, 3)


def get_texture_color(tex_coords: rs.texture_coordinate, texture_data, w, h, bytes_per_pixel, stride_in_bytes):
    """
    Returns a tuple in (R,G,B) format.
    """
    x = min(max(int(tex_coords[0] * w + 0.5), 0), w - 1)
    y = min(max(int(tex_coords[1] * h + 0.5), 0), h - 1)
    idx = x * bytes_per_pixel + y * stride_in_bytes
    return [texture_data[idx][0], texture_data[idx + 1][0], texture_data[idx + 2][0]]
