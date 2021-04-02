import numpy as np
import pyrealsense2 as rs

def export_numpy_array_to_ply(array, texture, file_name="test.ply"):
    # multiply y and z columns in order to rotate them to be displayed properly
    array[:, 1] *= -1
    array[:, 2] *= -1

    # concatenate both vertices and texture in one array with (length, 5) shape
    array = np.concatenate((array, texture), axis=1)

    # construct the header
    length = len(array)
    header = """ply
format ascii 1.0
comment pointcloud saved from PyCharm
element vertex {length}
property float32 x
property float32 y
property float32 z
property uchar blue
property uchar green
property uchar red
end_header""".format(length=length)

    np.savetxt(file_name, array, fmt="%.6g", header=header, delimiter=" ", comments="")


def default_export_points(points, file_name="default.ply"):
    # Create save_to_ply object
    ply = rs.save_to_ply(file_name)

    # Set options to the desired values
    # In this example we'll generate a textual PLY with normals (mesh is already created by default)
    ply.set_option(rs.save_to_ply.option_ply_binary, False)
    ply.set_option(rs.save_to_ply.option_ply_normals, True)
    ply.set_option(rs.save_to_ply.option_ignore_color, False)

    # Apply the processing block to the frameset which contains the depth frame and the texture
    ply.process(points)
    print(file_name + " saved.")