import pyrealsense2 as rs
import numpy as np
import cv2

from features.features import match_frames_FLANN_angles, compute_K_from_intrinsics, normalize, get_correspondences
from input_output.ply import export_numpy_array_to_ply
from mathematics.matrix import get_indexes_of_valid_points
from processing.icp import best_fit_transform
from processing.process import get_texture_for_pointcloud

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

# Processing blocks
pc = rs.pointcloud()
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 3)
colorizer = rs.colorizer()

profile=  cfg.get_stream(rs.stream.color)
intr_color = profile.as_video_stream_profile().get_intrinsics()
# compute intrinsics matrix
K = compute_K_from_intrinsics(intr_color)
K_inv = np.linalg.inv(K)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = cfg.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

index = 0
count = 0
I = np.eye(4, 4)
transf_matrices = []
vertices_array = []
texture_data_array = []
tex_coords_array = []
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

        mapped_frame, color_source = color_frame, color_image

        depth_frame = decimate.process(depth_frame)
        points = pc.calculate(depth_frame)
        pc.map_to(mapped_frame)

        # Pointcloud data to arrays
        v, t = points.get_vertices(), points.get_texture_coordinates()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
        texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

        if index == 0:
            # keep information to get color for each frame only once since they are all the same
            color_w, color_h = color_frame.get_width(), color_frame.get_height()
            bytes_per_pixel = color_frame.get_bytes_per_pixel()
            stride_in_bytes = color_frame.get_stride_in_bytes()

            # first frame
            prev_color_frame = color_image
            prev_depth_frame = depth_image

            index = index + 1
            continue

        if index % 10 == 0:
            valid_points = get_indexes_of_valid_points(verts)
            verts = verts[valid_points]

            texture_data = np.asanyarray(color_frame.get_data()).view(np.uint8).reshape(-1, 1)
            vertices_array.append(verts)
            texture_data_array.append(texture_data)
            tex_coords_array.append(texcoords[valid_points])
            try:
                # current frame is color_image
                pts_A, pts_B = get_correspondences(prev_color_frame, color_image)
                # at least three correspondences found
                # get depth points associated
                depth_prev = np.array([prev_depth_frame[y, x] for x, y in pts_A]) * depth_scale
                depth_curr = np.array([depth_image[y, x] for x, y in pts_B]) * depth_scale

                # normalize points
                pts_A = normalize(pts_A, K_inv)
                pts_B = normalize(pts_B, K_inv)

                # remove zero depth points from both
                valid_a = [depth_prev > 0]
                valid_b = [depth_curr > 0]

                # need valid correspondences in both point sets
                valid = valid_a and valid_b

                # create 3 dimensional array with x,y,z coordinates
                pts_A = np.column_stack((pts_A, depth_prev))[valid]
                pts_B = np.column_stack((pts_B, depth_curr))[valid]

                print("Matches: ", len(pts_A))
                T = best_fit_transform(pts_B, pts_A)
                print(T)
            except ValueError as e:
                print(e)
                T = I

            transf_matrices.append(T)
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
        if count == 15:
            break
finally:
    # Stop streaming
    pipeline.stop()

cv2.destroyAllWindows()
main_color = get_texture_for_pointcloud(vertices_array[0].copy(), tex_coords_array[0], texture_data_array[0], color_w,
                                        color_h,
                                        bytes_per_pixel, stride_in_bytes)
for i in range(1, len(vertices_array)):
    # get color for current point cloud
    current_color = get_texture_for_pointcloud(vertices_array[i].copy(), tex_coords_array[i], texture_data_array[i],
                                               color_w, color_h,
                                               bytes_per_pixel, stride_in_bytes)
    # append color to the main color array
    main_color = np.vstack((main_color, current_color))

main_pc = vertices_array[0]
index = 1
length = len(vertices_array)
# apply transformation to align each point cloud over the first point cloud
while index < length:
    current_pc = vertices_array[index]

    # append a column of ones to transform the current point cloud
    # in homogeneous coordinates to do matrix multiplication
    ones = np.ones((len(current_pc), 1))
    current_pc = np.append(current_pc, ones, axis=1)

    # transpose current point cloud to allow matrix multiplication
    current_pc = current_pc.T

    # apply all transformation matrices to current point cloud
    j = 1
    while j <= index:
        current_transf = transf_matrices[j]
        current_pc = current_transf @ current_pc
        j = j + 1

    # transpose current point cloud back to have info in xyz coordinates
    current_pc = current_pc.T
    # remove column used to allow matrix multiplication
    current_pc = np.delete(current_pc, 3, axis=1)

    # append current point cloud to the main point cloud
    main_pc = np.vstack((main_pc, current_pc))

    # move to the next point cloud
    index = index + 1

export_numpy_array_to_ply(main_pc, main_color, "./demo/result.ply")
