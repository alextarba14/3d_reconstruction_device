from input_output.ply import import_pointcloud_from_ply, export_numpy_array_to_ply
from mathematics.transformations import remove_points_far_away_from_centroid

if __name__ == "__main__":
    points, colors = import_pointcloud_from_ply("test.ply")
    cutoff = 2
    indices = remove_points_far_away_from_centroid(points, cutoff)
    export_numpy_array_to_ply(points[indices], colors[indices], f'removed_with_cutoff_{cutoff}.ply', rotate_columns=False)