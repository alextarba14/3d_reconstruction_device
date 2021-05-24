import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from skimage.measure import ransac
from skimage.transform import AffineTransform, EssentialMatrixTransform

from features.features import get_correspondences, add_ones
import pyrealsense2 as rs

from input_output.ply import import_point_cloud_from_ply, export_numpy_array_to_ply
from processing.icp import best_fit_transform, icp_point_to_point


def test_harris_corner_detector(file_name="./pictures/color_5.png"):
    img = cv2.imread(file_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    cv2.imshow('dst', img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


def test_sub_pixel(file_name="./pictures/color_5.png"):
    img = cv2.imread(file_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find Harris corners
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

    # Now draw them
    res = np.hstack((centroids, corners))
    res = np.int0(res)
    img[res[:, 1], res[:, 0]] = [0, 0, 255]
    img[res[:, 3], res[:, 2]] = [0, 255, 0]

    cv2.imshow('dst', img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


def test_good_features_to_track(file_name="./pictures/color_5.png"):
    img = cv2.imread(file_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
    corners = np.int0(corners)

    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 3, 255, -1)

    plt.imshow(img), plt.show()


def test_SIFT(file_name="./pictures/color_5.png"):
    img = cv2.imread(file_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # replaced with SIFT_create() since SIFT() led to errors
    sift = cv2.SIFT_create()
    # kp = sift.detect(gray, None)
    kp, des = sift.detectAndCompute(gray, None)
    # cv2.drawKeypoints(gray, kp, img)
    cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('dst', img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


def test_SURF(file_name="./pictures/color_5.png"):
    img = cv2.imread(file_name, 0)

    # Create SURF object. You can specify params here or later.
    # Here I set Hessian Threshold to 400
    surf = cv2.SURF(400)

    # Find keypoints and descriptors directly
    kp, des = surf.detectAndCompute(img, None)

    print(len(kp))


def test_ORB(file_name="./pictures/color_5.png"):
    img = cv2.imread(file_name, 0)

    # Initiate STAR detector
    orb = cv2.ORB_create()

    # find the keypoints with ORB
    kp = orb.detect(img, None)

    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)

    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(img, kp, img, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(img2), plt.show()


def feature_match_BF(file_name1="./pictures/color_5.png", file_name2="./pictures/color_10.png"):
    img1 = cv2.imread(file_name1, 0)  # queryImage
    img2 = cv2.imread(file_name2, 0)  # trainImage

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], outImg=None, flags=2)

    plt.imshow(img3), plt.show()


def feature_match_FLANN_angles(file_name1="./pictures2/color_1.png", file_name2="./pictures2/color_15.png"):
    img1 = cv2.imread(file_name1, 0)  # queryImage
    img2 = cv2.imread(file_name2, 0)  # trainImage

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    knn_matches = flann.knnMatch(des1, des2, k=2)

    # -- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.7
    good_matches = []
    angles = []
    points_xy = []
    for i, pair in enumerate(knn_matches):
        try:
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                # get point coordinates in left image
                point_img1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))
                # get point coordinates in right image regarding left image
                point_img2 = tuple(np.round(kp2[m.trainIdx].pt).astype(int) + np.array([img1.shape[1], 0]))
                # compute the angle of the line
                angles.append(math.atan2(point_img1[0] - point_img2[0], point_img1[1] - point_img2[1]))
                # append both points to array
                points_xy.append((point_img1, tuple(point_img2 - np.array([img1.shape[1], 0]))))
                good_matches.append(m)
        except (ValueError, IndexError):
            pass

    points = np.array(points_xy)
    pts_A, pts_B = points[:, 0, :], points[:, 1, :]
    angles = np.array(angles)
    mean = np.mean(angles)
    std_dev = np.std(angles)
    # exclude points outside the (-0.5*sigma, 0,5*sigma) interval
    valid = (angles > (mean - std_dev)) & (angles < (mean + std_dev))
    good_matches = np.array(good_matches)
    # keep only valid matches
    good_matches = good_matches[valid]

    # -- Draw matches
    img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
    cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:50], img_matches,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.imshow(img_matches), plt.show()


def feature_match_BF_KNN(file_name1="./pictures2/color_4.png", file_name2="./pictures2/color_10.png"):
    img1 = cv2.imread(file_name1, 0)  # queryImage
    img2 = cv2.imread(file_name2, 0)  # trainImage

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with SIFT
    # prev_kp, des1 = orb.detectAndCompute(img1, None)
    # curr_kp, des2 = orb.detectAndCompute(img2, None)
    prev_feats = cv2.goodFeaturesToTrack(
        img1,
        1000, qualityLevel=0.01, minDistance=7
    )
    prev_kp = [cv2.KeyPoint(*f[0], _size=3) for f in prev_feats]

    curr_feats = cv2.goodFeaturesToTrack(
        img2,
        1000, qualityLevel=0.01, minDistance=7
    )
    curr_kp = [cv2.KeyPoint(*f[0], _size=3) for f in curr_feats]

    prev_kp, des1 = orb.compute(img1, prev_kp)
    curr_kp, des2 = orb.compute(img2, curr_kp)
    # create BFMatcher object
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des1, des2, k=2)

    good_matches = []
    good = []
    for i, pair in enumerate(matches):
        try:
            m, n = pair
            if m.distance < 0.7 * n.distance:
                p_prev = tuple(prev_kp[m.queryIdx].pt)
                p_curr = tuple(curr_kp[m.trainIdx].pt)
                good.append([p_prev, p_curr])
                good_matches.append(m)
        except (ValueError, IndexError):
            pass
    good = np.asarray(good, dtype=int)

    # get points in previous and current image
    pts_A, pts_B = good[:, 0, :], good[:, 1, :]
    fx = fy = 377.65606689453125
    w = 640
    h = 480
    K = np.array([
        [fx, 0, w // 2],
        [0, fy, h // 2],
        [0, 0, 1]
    ])

    # pts_A = normalize(pts_A, K)
    # pts_B = normalize(pts_B, K)
    model_robust, inliers = ransac((pts_A, pts_B), AffineTransform, min_samples=3, residual_threshold=0.001,
                                   max_trials=200)

    good_matches = np.array(good_matches)[inliers]
    print("Matches BF: ", len(good_matches))

    img3 = cv2.drawMatches(img1, prev_kp, img2, curr_kp, good_matches, outImg=None, flags=2)
    plt.imshow(img3), plt.show()


def feature_match_FLANN_RANSAC(file_name1="./pictures/test_color_1.png", file_name2="./pictures/test_color_2.png"):
    prev_image = cv2.imread(file_name1, 0)  # queryImage
    curr_image = cv2.imread(file_name2, 0)  # trainImage

    prev_kp, curr_kp, good_matches = get_correspondences_flann(prev_image, curr_image)

    img3 = cv2.drawMatches(prev_image, prev_kp, curr_image, curr_kp, good_matches, outImg=None, flags=2)
    plt.imshow(img3), plt.show()


def get_correspondences_flann(prev_image, curr_image):
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
    good_matches = []
    good = []
    for i, pair in enumerate(knn_matches):
        try:
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                p_prev = tuple(prev_kp[m.queryIdx].pt)
                p_curr = tuple(curr_kp[m.trainIdx].pt)
                good.append([p_prev, p_curr])
                good_matches.append(m)
        except (ValueError, IndexError):
            pass
    good = np.asarray(good, dtype=int)

    # get points in previous and current image
    pts_A, pts_B = good[:, 0, :], good[:, 1, :]

    # find inliers when computing homography
    M, mask = cv2.findHomography(pts_A, pts_B, cv2.RANSAC, 0.001)
    # M, mask = cv2.estimateAffine2D(pts_A, pts_B, method=cv2.RANSAC,ransacReprojThreshold=0.001)

    # get inliers where the mask is 1
    inliers = np.ravel(mask) == 1

    print("Previous: ", pts_A[inliers])
    print("Current: ", pts_B[inliers])

    pts_A = pts_A[inliers]

    for i in range(len(pts_A)):
        x = pts_A[i][0]
        y = pts_A[i][1]
        cv2.circle(prev_image, (x, y), 3, 255, -1)

    plt.imshow(prev_image), plt.show()

    pts_B = pts_B[inliers]
    for i in range(len(pts_B)):
        x = pts_B[i][0]
        y = pts_B[i][1]
        cv2.rectangle(curr_image, (x-10, y-10),(x+10, y+10), 255, 3, -1)
    plt.imshow(curr_image), plt.show()

    return prev_kp, curr_kp, np.array(good_matches)[inliers]


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
cfg = pipeline.start(config)

profile = cfg.get_stream(rs.stream.color)
intr_color = profile.as_video_stream_profile().get_intrinsics()

if __name__ == "__main__":
    # test_harris_corner_detector()
    # test_sub_pixel()
    # test_good_features_to_track("./pictures/color_5.png")
    # test_good_features_to_track("./pictures/color_15.png")
    # test_SIFT()
    # test_ORB()
    import time

    start = time.time()
    # feature_match_BF()
    # feature_match_FLANN_angles()
    feature_match_FLANN_RANSAC()

    prev_image = cv2.imread("./pictures/test_color_1.png", 0)  # queryImage
    curr_image = cv2.imread("./pictures/test_color_2.png", 0)  # trainImage

    prev_depth = cv2.imread("./pictures/test_depth_1.png", 0)
    curr_depth = cv2.imread("./pictures/test_depth_2.png", 0)

    pts_A, pts_B = get_correspondences(prev_image, curr_image)

    # taken from the camera
    depth_scale = 0.0010000000474974513
    depth_prev = np.array([prev_depth[y, x] for x, y in pts_A]) * depth_scale
    depth_curr = np.array([curr_depth[y, x] for x, y in pts_B]) * depth_scale

    # remove zero depth points from both
    valid_a = [depth_prev > 0]
    valid_b = [depth_curr > 0]

    # need valid correspondences in both point sets
    valid = valid_a and valid_b
    depth_prev = depth_prev[valid]
    depth_curr = depth_curr[valid]

    pts_A = pts_A[valid]
    pts_B = pts_B[valid]

    left_points = []
    right_points = []
    i = 0
    length = len(pts_A)
    while i < length:
        left_points.append(rs.rs2_deproject_pixel_to_point(intr_color, pts_A[i], depth_prev[i]))
        right_points.append(rs.rs2_deproject_pixel_to_point(intr_color, pts_B[i], depth_curr[i]))
        i = i + 1

    left_points = np.array(left_points)
    right_points = np.array(right_points)

    T = icp_point_to_point(right_points, left_points, tolerance=0.000001)

    print("Transformation: ", T)
    points_a, colors_a = import_point_cloud_from_ply("./pictures/point_cloud_1.ply")
    points_b, colors_b = import_point_cloud_from_ply("./pictures/point_cloud_2.ply")

    points_a_ext = add_ones(points_a)
    T = np.linalg.inv(T)
    result = (T @ points_a_ext.T)[:3].T
    export_numpy_array_to_ply(result, colors_a, "./pictures/test_result_1_2.ply", rotate_columns=False)

    print("It took:", time.time() - start)
