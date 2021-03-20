import numpy as np


def export_numpy_array_to_ply(array, file_name="test.ply"):
    # transpose numpy array in order to put columns as lines
    transposed_array = array.transpose()
    # get the z column - depth
    transposed_depth = transposed_array[2]
    # keep only elements that have depth greater than 0
    indexes = np.where(transposed_depth>0)
    array = array[indexes]

    # construct the header
    length = len(array)
    header = """ply
format ascii 1.0
comment pointcloud saved from PyCharm
element vertex {length}
property float32 x
property float32 y
property float32 z
end_header""".format(length=length)

    np.savetxt(file_name, array, fmt="%.6g", header=header, delimiter=" ", comments="")
