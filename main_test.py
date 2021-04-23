from input_output.ply import import_pointcloud_from_ply, export_numpy_array_to_ply
from mathematics.transformations import remove_points_far_away_from_centroid, remove_points_with_less_neighbours
import numpy as np

if __name__ == "__main__":
    points, colors = import_pointcloud_from_ply("test.ply")
    # cutoff = 2
    # indices = remove_points_far_away_from_centroid(points, cutoff)
    # export_numpy_array_to_ply(points[indices], colors[indices], f'removed_with_cutoff_{cutoff}.ply', rotate_columns=False)

    nb_neighbours = 35
    radius = 0.05
    indices = remove_points_with_less_neighbours(points, nb_neighbours, radius)
    colors[indices] = (0, 255, 0)
    export_numpy_array_to_ply(points[indices], colors[indices], f'removed_neighbours_less_than_{nb_neighbours}_radius_{radius}.ply',
                              rotate_columns=False)

    indices = np.invert(indices)
    colors[indices] = (0, 0, 255)
    export_numpy_array_to_ply(points[indices], colors[indices], f'what_has_been_removed_{nb_neighbours}_radius_{radius}.ply',
                              rotate_columns=False)