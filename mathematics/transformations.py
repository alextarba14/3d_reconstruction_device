import numpy as np
import scipy.spatial as spatial
import time

def remove_points_far_away_from_centroid(points, cutoff=1.0):
    # find the points` centroid
    centroid = np.mean(points, axis=0)
    distances = np.empty(len(points))
    for i in range(len(points)):
        # exclude points that have zero depth
        if points[i][2] != 0:
            dist = np.linalg.norm(centroid - points[i])
            distances[i] = dist

    # keep only points that are under the cutoff value
    return distances < cutoff


def remove_points_with_less_neighbours(points, nb_neighbours, radius=0.03):
    tree = spatial.cKDTree(points)
    neighbours = np.zeros(len(points))
    for i in range(len(points)):
        # exclude points that have zero depth
        if points[i][2] != 0:
            if neighbours[i] == 0:
                nearest = tree.query_ball_point(points[i], radius)
                if len(nearest) > nb_neighbours:
                    neighbours[nearest] = 1

    # keep only points that have no. of neighbours above the threshold
    return  neighbours == 1


def apply_transformations(pointclouds, transf_matrices):
    """
    Apply transformation to each pointcloud by multiplying with their correspondent transformation matrix.
    P' = P*Tr_matrix
    Each pointcloud has a transformation matrix associated that represents the transformations that has been done
    until the current pointcloud. So there is no need to multiply the pointcloud with it's transformation matrix once more.
    It will multiply each pointcloud with the future transformations and update the pointclouds.
    """
    start_time_function = time.time()
    print("apply_transformations started at: ", start_time_function)

    length = len(pointclouds)
    index = 0
    while index < length:
        start_time_index = time.time()
        print(f'Started index: {index} at: ', start_time_index)
        current_pointcloud = pointclouds[index]
        pointclouds[index] = apply_next_transformations_to_current_pointcloud(current_pointcloud, index,
                                                                              transf_matrices, length)

        print(f'Stopped index: {index} after: ', time.time() - start_time_index)
        index = index + 1

    print("Ended after: ", time.time() - start_time_function)
    return pointclouds


def apply_next_transformations_to_current_pointcloud(current_pointcloud, index, transf_matrices, length):
    """
    Apply all future transformations that have happened after the current pointcloud.
    Multiply current_pointcloud[i] with all transformation matrices from [i+1, length).
    Each transformation matrix is in shape(4,4), but a point from current_pointcloud is in (1,3) shape.
    A column full of ones has been appended to the pointcloud to allow multiplication.
    current_pointcloud(1,4) *  transf_matrix(4,4) = pointcloud(1,4), the result will be in the (x,y,z,1) format
    after multiplications the end column will be removed.
    """
    # return current pointcloud when it is the last pointcloud
    if (index + 1) == length:
        return current_pointcloud

    # append a column full of ones at the end
    pc_length = len(current_pointcloud)
    ones = np.ones((pc_length, 1), dtype=np.float32)
    current_pointcloud = np.append(current_pointcloud, ones, axis=1)

    # apply transformations for each pointcloud
    j = index + 1
    while j < length:
        start_time = time.time()
        print(f'Started j: {j} at: ', start_time)
        current_transf_matrix = transf_matrices[j]

        # for i in range(pc_length):
        #     # multiply Tr*p = p' (obtaining points based referenced at previous system information)
        #     current_pointcloud[i] = current_transf_matrix.dot(current_pointcloud[i])
        # or
        # current_pointcloud = np.einsum("ij,kj->ik", current_pointcloud, current_transf_matrix)
        # or
        # current_pointcloud = current_pointcloud.dot(current_transf_matrix)
        current_pointcloud = current_pointcloud @ current_transf_matrix
        j = j + 1
        print(f'Stopped j: {j} after: ', time.time() - start_time)

    # removing the last column since it was added to perform dot product between vector[1x(3+1)] and transform matrix[4x4]
    current_pointcloud = np.delete(current_pointcloud, 3, axis=1)
    return current_pointcloud

