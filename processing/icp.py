import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import ARDRegression
import random
import time

def best_fit_transform(A, B):
    """
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: nxm numpy array of corresponding points
      B: nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    """

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # find centroids of each point clouds
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # translate points to their centroids
    AA = A - centroid_A
    BB = B - centroid_B

    # get rotation matrix from SVD
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # compute the transformation matrix in homogeneous coordinates
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    """
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: nxm array of points: source
        dst: nxm array of points: destination
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbors
    """

    assert src.shape == dst.shape

    # algorithm: {'auto', 'ball_tree', 'kd_tree', 'brute'
    # ball_tree -> best_fit: 0.0188800121296459
    # kd_tree -> best_fit: 0.018880012129646167
    # 0.01887971474324085
    neigh = NearestNeighbors(n_neighbors=1, radius=0.01, leaf_size=40, n_jobs=-1, p=2)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=30, tolerance=0.001):
    """
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: nxm numpy array of source mD points
        B: nxm numpy array of destination mD points
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    """

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    best_fit = np.inf
    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)

        # compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform(src[:m, :].T, dst[:m, indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if mean_error < tolerance:
            break
        if best_fit > mean_error:
            best_fit = mean_error
            copy_src = src.copy()
            copy_T = T.copy()
            print("Best_fit: ", best_fit)
        print(f'Step {i} from {max_iterations}...')

    # calculate final transformation
    T, _, _ = best_fit_transform(A, src[:m, :].T)

    return copy_T, distances, i, copy_src.T
