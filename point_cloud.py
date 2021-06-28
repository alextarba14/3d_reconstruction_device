"""
OpenCV and Numpy Point cloud Software Renderer
------
Keyboard:
    [c]     Display only color or color and depth map.
    [d]     Cycle through decimation values
    [e]     Export points to ply (./<unix_timestamp>.ply)
    [q\ESC] Quit
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import time

from input_output.ply import default_export_points, export_numpy_array_to_ply
from mathematics.matrix import get_indexes_of_valid_points, get_trapz_integral_by_time, create_rotation_matrix, \
    create_transformation_matrix
from mathematics.transformations import apply_transformations
from processing.process import remove_statistical_outliers, down_sample_point_cloud, \
    get_color_from_image


class AppState:
    def __init__(self):
        self.WIN_NAME = 'RealSense'
        self.decimate = 2
        self.color = True


state = AppState()

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

# Start streaming color & depth
pipeline.start(config)

# Create a different pipeline for IMU
imu_pipeline = rs.pipeline()
imu_config = rs.config()
# Configuring streams at different rates
# Accelerometer available FPS: {63, 250}Hz
# imu_config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)  # acceleration
# Gyroscope available FPS: {200,400}Hz
imu_config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)  # gyroscope

# Start streaming IMU
imu_profile = imu_pipeline.start(imu_config)

# Get stream profile and camera intrinsics
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

# align depth to color frames
align_to = rs.stream.color
align = rs.align(align_to)

# Processing blocks
pc = rs.pointcloud()
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 3)
colorizer = rs.colorizer()

cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow(state.WIN_NAME, w, h)

vertices_array = []
color_array = []

threshold = 10
frame_count = -1
gyro_data_array = [[0 for x in range(3)] for y in range(threshold)]
transf_matrices = []
index = 0
mat_count = 0
while True:
    # Render
    now = time.time()

    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()

    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    frame_count = frame_count + 1
    # Wait for a coherent pair of frames: gyro and accel
    imu_frames = imu_pipeline.wait_for_frames()

    for frame in imu_frames:
        motion_frame = frame.as_motion_frame()
        # Get the timestamp of the current frame to integrate by time
        timestamp = motion_frame.get_timestamp()
        if motion_frame and motion_frame.get_profile().stream_type() == rs.stream.gyro:
            # Gyro frame
            # Get gyro measurements
            gyro_data = motion_frame.get_motion_data()
            gyro_data_array[index] = [gyro_data.x, gyro_data.y, gyro_data.z, timestamp]

    # create a depth color map for visualization
    depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    depth_frame = decimate.process(depth_frame)

    # Grab new intrinsics (may be changed by decimation)
    depth_intrinsics = rs.video_stream_profile(
        depth_frame.profile).get_intrinsics()
    w, h = depth_intrinsics.width, depth_intrinsics.height

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    mapped_frame, color_source = color_frame, color_image

    points = pc.calculate(depth_frame)
    pc.map_to(mapped_frame)

    # Pointcloud data to arrays
    v, t = points.get_vertices(), points.get_texture_coordinates()
    verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
    texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

    if frame_count == 0:
        continue

    index = frame_count % threshold
    if index == 0:
        print("Frame count:", mat_count)
        rotation = get_trapz_integral_by_time(gyro_data_array)
        rotation_matrix = create_rotation_matrix(rotation)

        # multiply rotation matrix with translation matrix in homogeneous coordinates
        transf_mat = create_transformation_matrix(rotation_matrix, [0, 0, 0])
        # keep only valid points in point cloud
        valid_points = get_indexes_of_valid_points(verts)
        # get color from image based on texture coordinates
        color = get_color_from_image(texcoords, color_image)
        vertices_array.append(verts[valid_points])
        color_array.append(color[valid_points])
        transf_matrices.append(transf_mat)

        mat_count = mat_count + 1

    if mat_count == 10:
        # continue with the processing part
        break

    dt = time.time() - now

    cv2.setWindowTitle(
        state.WIN_NAME, "RealSense (%dx%d) %dFPS (%.2fms) %s" %
                        (w, h, 1.0 / dt, dt * 1000, ""))

    if state.color:
        cv2.imshow(state.WIN_NAME, color_image)
    else:
        cv2.imshow(state.WIN_NAME, np.hstack((color_image, depth_colormap)))

    # wait for keyboard interruptsc
    key = cv2.waitKey(1)
    if key == ord("c"):
        state.color = not state.color

    if key == ord("d"):
        state.decimate = (state.decimate + 1) % 3
        decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)

    if key == ord("e"):
        texture = get_color_from_image(texcoords, color_image)
        export_numpy_array_to_ply(verts, texture, f'{time.time()}.ply')

    if key in (27, ord("q")) or cv2.getWindowProperty(state.WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
        break

# Stop streaming
pipeline.stop()
start_time = time.time()
# close opencv window
cv2.destroyAllWindows()
print("Applying transformation matrices...")
updated_vertices_array = apply_transformations(vertices_array, transf_matrices)
print("Removing outliers from point clouds...")
colors_list = []
for i in range(len(updated_vertices_array)):
    # remove outliers statistically from each point cloud
    valid_indices = remove_statistical_outliers(updated_vertices_array[i], nb_neighbours=50, std_ratio=1)
    updated_vertices_array[i] = updated_vertices_array[i][valid_indices]
    # append color to the main color array with valid indices
    colors_list.append(color_array[i][valid_indices])
    # append to bigger array
    if i == 0:
        main_pc = updated_vertices_array[0]
        continue
    main_pc = np.vstack((main_pc, updated_vertices_array[i]))

print("Outliers removed.")

# create main_color array to match main_point_cloud
main_color = np.vstack(colors_list)

print("Down sampling point cloud.")
indices = down_sample_point_cloud(main_pc)
print("Without exporting: ", (time.time() - start_time))
export_numpy_array_to_ply(main_pc, main_color, f"./demo/result_{time.time()}.ply")

export_numpy_array_to_ply(main_pc[indices], main_color[indices], "./demo/down_sampled_result.ply", rotate_columns=False)
removed_indices = np.invert(indices)

export_numpy_array_to_ply(main_pc[removed_indices], main_color[removed_indices], "./demo/removed_from_result.ply",
                          rotate_columns=False)