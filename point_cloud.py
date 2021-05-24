# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

"""
OpenCV and Numpy Point cloud Software Renderer
------
Keyboard:
    [d]     Cycle through decimation values
    [e]     Export points to ply (./out.ply)
    [q\ESC] Quit
"""

import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs
import time
from input_output.ply import default_export_points, export_numpy_array_to_ply
from mathematics.matrix import get_trapz_integral_by_time, create_rotation_matrix, create_transformation_matrix, \
    get_indexes_of_valid_points
from mathematics.transformations import apply_transformations
from processing.icp import icp
from processing.process import get_texture_from_pointcloud, remove_statistical_outliers, down_sample_point_cloud


class AppState:

    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 2
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)


state = AppState()

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, rs.format.z16, 30)
config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

# Create a different pipeline for IMU
imu_pipeline = rs.pipeline()
imu_config = rs.config()
# Configuring streams at different rates
# Accelerometer available FPS: {63, 250}Hz
imu_config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)  # acceleration
# Gyroscope available FPS: {200,400}Hz
imu_config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)  # gyroscope

# Start streaming IMU
imu_profile = imu_pipeline.start(imu_config)

# Start streaming color & depth
pipeline.start(config)

# Get stream profile and camera intrinsics
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

# Processing blocks
pc = rs.pointcloud()
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 3)
colorizer = rs.colorizer()

cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow(state.WIN_NAME, 640,480)


def project(v):
    """project 3d vector array to 2d"""
    h, w = out.shape[:2]
    view_aspect = float(h)/w

    # ignore divide by zero for invalid depth
    with np.errstate(divide='ignore', invalid='ignore'):
        proj = v[:, :-1] / v[:, -1, np.newaxis] * \
            (w*view_aspect, h) + (w/2.0, h/2.0)

    # near clipping
    znear = 0.03
    proj[v[:, 2] < znear] = np.nan
    return proj


def view(v):
    """apply view transformation on vector array"""
    return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation


def pointcloud(out, verts, texcoords, color, painter=True):
    """draw point cloud with optional painter's algorithm"""
    if painter:
        # Painter's algo, sort points from back to front

        # get reverse sorted indices by z (in view-space)
        # https://gist.github.com/stevenvo/e3dad127598842459b68
        v = view(verts)
        s = v[:, 2].argsort()[::-1]
        proj = project(v[s])
    else:
        proj = project(view(verts))

    if state.scale:
        proj *= 0.5**state.decimate

    h, w = out.shape[:2]

    # proj now contains 2d image coordinates
    j, i = proj.astype(np.uint32).T

    # create a mask to ignore out-of-bound indices
    im = (i >= 0) & (i < h)
    jm = (j >= 0) & (j < w)
    m = im & jm

    cw, ch = color.shape[:2][::-1]
    if painter:
        # sort texcoord with same indices as above
        # texcoords are [0..1] and relative to top-left pixel corner,
        # multiply by size and add 0.5 to center
        v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
    else:
        v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
    # clip texcoords to image
    np.clip(u, 0, ch-1, out=u)
    np.clip(v, 0, cw-1, out=v)

    # perform uv-mapping
    out[i[m], j[m]] = color[u[m], v[m]]


out = np.empty((h, w, 3), dtype=np.uint8)

vertices_array = []
tex_coords_array = []
color_frames = []
transf_matrices = []

threshold = 10
frame_count = -1
accel_state = [0, -9.81, 0, 1]
accel_data_array = [[0 for x in range(3)] for y in range(threshold)]
gyro_data_array = [[0 for x in range(3)] for y in range(threshold)]

index = 0
mat_count = -1
while True:
    # Grab camera data
    if not state.paused:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        frame_count = frame_count + 1

        # Wait for a coherent pair of frames: gyro and accel
        imu_frames = imu_pipeline.wait_for_frames()

        for frame in imu_frames:
            motion_frame = frame.as_motion_frame()
            # Get the timestamp of the current frame to integrate by time
            timestamp = motion_frame.get_timestamp()
            if motion_frame and motion_frame.get_profile().stream_type() == rs.stream.accel:
                # Accelerometer frame
                # Get accelerometer measurements
                accel_data = motion_frame.get_motion_data()
                accel_data_array[index] = [accel_data.x, accel_data.y, accel_data.z, timestamp]
                # accel_data_array[index] = get_difference_item(accel_data_array, accel_data, index)
                # accel_data_array[index].append(timestamp)
                # print(accel_data)
            elif motion_frame and motion_frame.get_profile().stream_type() == rs.stream.gyro:
                # Gyro frame
                # Get gyro measurements
                gyro_data = motion_frame.get_motion_data()
                gyro_data_array[index] = [gyro_data.x, gyro_data.y, gyro_data.z, timestamp]

        depth_frame = decimate.process(depth_frame)

        # Grab new intrinsics (may be changed by decimation)
        depth_intrinsics = rs.video_stream_profile(
            depth_frame.profile).get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_colormap = np.asanyarray(
            colorizer.colorize(depth_frame).get_data())

        if state.color:
            mapped_frame, color_source = color_frame, color_image
        else:
            mapped_frame, color_source = depth_frame, depth_colormap

        points = pc.calculate(depth_frame)
        pc.map_to(mapped_frame)

        # Pointcloud data to arrays
        v, t = points.get_vertices(), points.get_texture_coordinates()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
        texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

        if frame_count == 0:
            # reject the first frame since it is has very dark texture
            continue

        index = frame_count % threshold
        if index == 0:
            print("Frame count:", frame_count)
            # get the rotation angle by integrating the rotation velocity
            rotation = get_trapz_integral_by_time(gyro_data_array)
            rotation_matrix = create_rotation_matrix(rotation)

            # multiply rotation matrix with translation matrix in homogeneous coordinates
            transf_mat = create_transformation_matrix(rotation_matrix, [0,0,0])

            # append to bigger lists
            valid_points = get_indexes_of_valid_points(verts)
            vertices_array.append(verts[valid_points])
            transf_matrices.append(transf_mat)
            color_frames.append(color_frame)
            tex_coords_array.append(texcoords[valid_points])
            mat_count = mat_count + 1

        if mat_count == 10:
            # continue with the processing part
            break

    # Render
    now = time.time()

    out.fill(0)

    # grid(out, (0, 0.5, 1), size=1, n=10)
    # frustum(out, depth_intrinsics)
    # axes(out, view([0, 0, 0]), state.rotation, size=0.1, thickness=1)

    # if not state.scale or out.shape[:2] == (h, w):
    #     pointcloud(out, verts, texcoords, color_source)
    # else:
    #     tmp = np.zeros((h, w, 3), dtype=np.uint8)
    #     pointcloud(tmp, verts, texcoords, color_source)
    #     tmp = cv2.resize(
    #         tmp, out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    #     np.putmask(out, tmp > 0, tmp)

    dt = time.time() - now

    cv2.setWindowTitle(
        state.WIN_NAME, "RealSense (%dx%d) %dFPS (%.2fms) %s" %
        (w, h, 1.0/dt, dt*1000, "PAUSED" if state.paused else ""))

    cv2.imshow(state.WIN_NAME, out)
    key = cv2.waitKey(1)

    if key == ord("d"):
        state.decimate = (state.decimate + 1) % 3
        decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)

    if key == ord("e"):
        # points.export_to_ply('./out.ply', mapped_frame)
        # default_export_points(points)
        texture = get_texture_from_pointcloud(verts, texcoords, color_frame)
        export_numpy_array_to_ply(verts, texture, f'{time.time()}.ply')

    if key in (27, ord("q")) or cv2.getWindowProperty(state.WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
        break

# Stop streaming
pipeline.stop()

# apply transformation to each point cloud
updated_pointclouds = apply_transformations(vertices_array, transf_matrices)

print("Removing outliers from point clouds...")
for i in range(len(vertices_array)):
    # remove outliers statistically from each point cloud
    valid_indices = remove_statistical_outliers(updated_pointclouds[i], nb_neighbours=50, std_ratio=1)
    updated_pointclouds[i] = updated_pointclouds[i][valid_indices]

    # update the color data as well
    tex_coords_array[i] = tex_coords_array[i][valid_indices]

print("Outliers removed.")

# create a main point cloud with associated color
main_pc = updated_pointclouds[0]
main_color = get_texture_from_pointcloud(updated_pointclouds[0], tex_coords_array[0], color_frames[0])

# parse each point cloud to get the color
for i in range(1, len(updated_pointclouds)):
    current_color = get_texture_from_pointcloud(updated_pointclouds[i], tex_coords_array[i], color_frames[i])

    # append color to the main color array
    main_color = np.vstack((main_color, current_color))
    # append current point cloud to the main point cloud
    main_pc = np.vstack((main_pc, updated_pointclouds[i]))

print("Down sampling point cloud.")
indices = down_sample_point_cloud(main_pc)

export_numpy_array_to_ply(main_pc, main_color, "./demo/result.ply")


export_numpy_array_to_ply(main_pc[indices], main_color[indices], "./demo/down_sampled_result.ply", rotate_columns=False)
removed_indices = np.invert(indices)

export_numpy_array_to_ply(main_pc[removed_indices], main_color[removed_indices], "./demo/removed_from_result.ply", rotate_columns=False)


