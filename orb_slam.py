import pyrealsense2 as rs
import numpy as np
import cv2

from features.features import match_frames_FLANN_angles, compute_K_from_intrinsics, normalize
from processing.icp import best_fit_transform

pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

# Configure depth and color streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
cfg = pipeline.start(config)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

profile = cfg.get_stream(rs.stream.depth)
intrinsics = profile.as_video_stream_profile().get_intrinsics()
# compute intrinsics matrix
K = compute_K_from_intrinsics(intrinsics)
K_inv = np.linalg.inv(K)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = cfg.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

index = 0
count = 0
try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        if index % 5 == 0:
            if count == 0:
                # first frame
                prev_color_frame = color_image
                prev_depth_frame = depth_image
            else:
                # current frame is color_image
                try:
                    pts_A, pts_B = match_frames_FLANN_angles(prev_color_frame, color_image)
                    # at least three correspondences found
                    # get depth points associated
                    depth_prev = np.array([prev_depth_frame[y, x] for x, y in pts_A]) * depth_scale
                    depth_curr = np.array([depth_image[y, x] for x, y in pts_B]) * depth_scale

                    # normalize points
                    pts_A = normalize(pts_A, K_inv)
                    pts_B = normalize(pts_B, K_inv)

                    # create 3 dimensional array with x,y,z coordinates

                    print("Before removing: ", len(pts_A))

                    # remove zero depth points from both
                    valid_a = [depth_prev > 0]
                    valid_b = [depth_curr > 0]

                    # need valid correspondences in both point sets
                    valid = valid_a and valid_b

                    pts_A = np.column_stack((pts_A, depth_prev))[valid]
                    pts_B = np.column_stack((pts_B, depth_curr))[valid]

                    print("Matches: ", len(pts_A))
                    T = best_fit_transform(pts_A, pts_B)
                    print(T)
                except ValueError as e:
                    print(e)

            prev_color_frame = color_image
            prev_depth_frame = depth_image
            count = count + 1

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        key = cv2.waitKey(1)

        if key in (27, ord("q")):
            break

        index = index + 1
finally:
    # Stop streaming
    pipeline.stop()

