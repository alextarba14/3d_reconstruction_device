import numpy as np
import pyrealsense2 as rs


def export_numpy_array_to_ply(verts, texture, file_name="test.ply", rotate_columns=True):
    # multiply y and z columns in order to rotate them to be displayed properly
    if rotate_columns:
        verts[:, 1] *= -1
        verts[:, 2] *= -1

    # concatenate both vertices and texture in one array with (length, 5) shape
    verts = np.concatenate((verts, texture), axis=1)

    # construct the header
    length = len(verts)
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

    np.savetxt(file_name, verts, fmt="%.6g", header=header, delimiter=" ", comments="")


def default_export_points(points, file_name="default.ply"):
    # Create save_to_ply object
    ply = rs.save_to_ply(file_name)

    # Set options to the desired values
    # In this example we'll generate a textual PLY with normals (mesh is already created by default)
    ply.set_option(rs.save_to_ply.option_ply_binary, False)
    ply.set_option(rs.save_to_ply.option_ply_normals, False)
    ply.set_option(rs.save_to_ply.option_ignore_color, False)

    # Apply the processing block to the frameset which contains the depth frame and the texture
    ply.process(points)
    print(file_name + " saved.")


def import_pointcloud_from_ply(file_name="test.ply"):
    """
    Imports a point cloud from a .ply file that contains xyz coordinates and color for each point.
    The coordinates are in the first 3 columns and the colors in the last 3 columns of data.

    Returns:
        points: points' coordinates with xyz values
        colors: colors associated with each point
    """
    # skip the header which is 11 lines long
    result = np.loadtxt(file_name, delimiter=" ", dtype=np.float32, skiprows=11)

    # get the x,y,z coordinates from file
    points = result[:, :3]

    # get the color associated with each point as values in [0,255]
    colors = result[:, 3:].astype(np.uint8)
    return points, colors

