import cv2
import numpy as np
import math


def match_frames_FLANN_angles(prev_image, curr_image):
    """
    Extract key points and compute descriptors using ORB.
    For each key point in the previous image search for two matches in the current image.
    Then filter matches using the Lowe's ratio test and compute angles for the correct ones.
    Based on the angles compute the mean and standard deviation and exclude points that are
    not in a given interval regarding both of them.
    If number of correspondences is less than 3 a ValueError is thrown since at least 3 points
    are needed to compute the Transformation matrix(Rotation and translation).

    Args:
        prev_image: a numpy array containing RGB of the previous frame.
        curr_image: a numpy array containing RGB of the current frame.
    Returns:
        valid_pts_A: a numpy array with points that are considered valid from prev_image.
        valid_pts_B: a numpy array with points that are considered valid from curr_image.
    """
    # Initiate ORB detector
    orb = cv2.ORB_create()

    # find the key points and descriptors with ORB
    prev_kp, prev_des = orb.detectAndCompute(prev_image, None)
    curr_kp, curr_des = orb.detectAndCompute(curr_image, None)

    # FLANN parameters
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    knn_matches = flann.knnMatch(prev_des, curr_des, k=2)

    # -- Filter matches using the Lowe's ratio test
    angles = []
    points_xy = []
    for i, pair in enumerate(knn_matches):
        try:
            m, n = pair
            if m.distance < 0.7 * n.distance:
                # get point coordinates in left image
                point_img1 = tuple(np.round(prev_kp[m.queryIdx].pt).astype(int))
                # get point coordinates in right image regarding left image
                point_img2 = tuple(np.round(curr_kp[m.trainIdx].pt).astype(int) + np.array([prev_image.shape[1], 0]))
                # compute the angle of the line
                angles.append(math.atan2(point_img1[0] - point_img2[0], point_img1[1] - point_img2[1]))
                # append both points to array
                points_xy.append((point_img1, tuple(point_img2 - np.array([prev_image.shape[1], 0]))))
        except (ValueError, IndexError):
            pass

    points = np.array(points_xy)

    # at least three correspondence are needed to compute Transformation matrix
    if len(points) < 3:
        raise ValueError("No correspondences found.")

    # extract points in the previous and current image
    pts_A, pts_B = points[:, 0, :], points[:, 1, :]

    angles = np.array(angles)
    mean = np.mean(angles)
    std_dev = np.std(angles)
    # exclude points outside the (mean - 0.5 *sigma, mean + 0.5 * sigma) interval
    valid = (angles > (mean - 0.5 * std_dev)) & (angles < (mean + 0.5 * std_dev))

    valid_pts_A = pts_A[valid]
    valid_pts_B = pts_B[valid]

    # at least three correspondence are needed to compute Transformation matrix
    if len(valid_pts_A) < 3:
        raise ValueError("No correspondences found.")

    return valid_pts_A, valid_pts_B


def get_correspondences(prev_image, curr_image):
    """
    Get correspondences between frames using keypoint features and descriptors matched by ORB.
    Using OpenCV findHomography with RANSAC separate inliers from outliers.
    It returns only the inliers points in both frames.
    Args:
        prev_image: a numpy array containing RGB of the previous frame.
        curr_image: a numpy array containing RGB of the current frame.
    Returns:
        pts_A[inliers]: a numpy array with points that are considered valid from prev_image.
        pts_B[inliers]: a numpy array with points that are considered valid from curr_image.
    """
    # Initiate ORB detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with ORB
    prev_kp, prev_des = orb.detectAndCompute(prev_image, None)
    curr_kp, curr_des = orb.detectAndCompute(curr_image, None)

    # FLANN parameters
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    knn_matches = flann.knnMatch(prev_des, curr_des, k=2)

    # -- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.7
    good = []
    for i, pair in enumerate(knn_matches):
        try:
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                p_prev = tuple(prev_kp[m.queryIdx].pt)
                p_curr = tuple(curr_kp[m.trainIdx].pt)
                good.append([p_prev, p_curr])
        except (ValueError, IndexError):
            pass
    good = np.asarray(good, dtype=int)

    # get points in previous and current image
    pts_A, pts_B = good[:, 0, :], good[:, 1, :]

    # find inliers by computing homography
    try:
        M, mask = cv2.findHomography(pts_A, pts_B, cv2.RANSAC, 0.001)
    except cv2.error:
        raise ValueError("Couldn't find correspondences")

    # get inliers indices
    inliers = np.ravel(mask) == 1
    return pts_A[inliers], pts_B[inliers]


def add_ones(arr):
    """
    Append a a column full of ones to transform the given array to homogeneous coordinates.
    """
    return np.concatenate([arr, np.ones((arr.shape[0], 1))], axis=1)


def normalize(pts, K_inv):
    """
    Convert from RGB coordinates to camera coordinates.
    """
    pts = add_ones(pts)
    return (K_inv @ pts.T).T[:, :2]


def compute_K_from_intrinsics(intrinsics):
    """
    Given the intrinsics compute the camera matrix.
    """
    fx = intrinsics.fx
    fy = intrinsics.fy
    w = intrinsics.width
    h = intrinsics.height
    return np.array([
        [fx, 0, w // 2],
        [0, fy, h // 2],
        [0, 0, 1]
    ])
