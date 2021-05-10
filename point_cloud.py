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

from icp_point_to_plane.icp_point_to_plane import icp
from input_output.ply import default_export_points, export_numpy_array_to_ply
from mathematics.matrix import get_indexes_of_valid_points
from processing.process import get_texture_for_pointcloud


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
cv2.resizeWindow(state.WIN_NAME, w, h)

out = np.empty((h, w, 3), dtype=np.uint8)

vertices_array = []
tex_coords_array = []
texture_data_array = []
transf_matrices = []

threshold = 10
frame_count = -1

index = 0
mat_count = 0
while True:
    # Grab camera data
    if not state.paused:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        frame_count = frame_count + 1

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
            # keep information to get color for each frame only once since they are all the same
            color_w, color_h = color_frame.get_width(), color_frame.get_height()
            bytes_per_pixel = color_frame.get_bytes_per_pixel()
            stride_in_bytes = color_frame.get_stride_in_bytes()
            continue

        index = frame_count % threshold
        if index == 0:
            print("Frame count:", frame_count)
            # get the rotation angle by integrating the rotation velocity
            # rotation = get_trapz_integral_by_time(gyro_data_array)
            # rotation_matrix = create_rotation_matrix(rotation)

            # multiply rotation matrix with translation matrix in homogeneous coordinates
            # transf_mat = create_transformation_matrix(rotation_matrix, [0,0,0])

            # keep only valid points in point cloud
            valid_points = get_indexes_of_valid_points(verts)
            verts = verts[valid_points]

            texture_data = np.asanyarray(color_frame.get_data()).view(np.uint8).reshape(-1, 1)
            vertices_array.append(verts)
            # transf_matrices.append(transf_mat)
            texture_data_array.append(texture_data)
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
                        (w, h, 1.0 / dt, dt * 1000, "PAUSED" if state.paused else ""))

    cv2.imshow(state.WIN_NAME, out)
    key = cv2.waitKey(1)

    if key == ord("d"):
        state.decimate = (state.decimate + 1) % 3
        decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)

    if key == ord("e"):
        # points.export_to_ply('./out.ply', mapped_frame)
        # default_export_points(points)
        texture = get_texture_for_pointcloud(verts, texcoords, color_frame)
        export_numpy_array_to_ply(verts, texture, f'{time.time()}.ply')

    if key in (27, ord("q")) or cv2.getWindowProperty(state.WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
        break

# Stop streaming
pipeline.stop()

colors = []
X_src = vertices_array[0].copy()
main_color = get_texture_for_pointcloud(X_src, tex_coords_array[0], texture_data_array[0], color_w, color_h,
                                        bytes_per_pixel, stride_in_bytes)
colors.append(main_color)
transf_matrices.append(np.eye(4, 4))

for i in range(1, len(vertices_array)):
    X_dst = vertices_array[i].copy()
    # get transformation matrix that match destination over source
    Tr = icp(X_src, X_dst)
    transf_matrices.append(Tr)
    # get color for current point cloud
    current_color = get_texture_for_pointcloud(X_dst, tex_coords_array[i], texture_data_array[i], color_w, color_h,
                                               bytes_per_pixel, stride_in_bytes)
    # append color to colors array
    colors.append(current_color)

    # update the current source
    X_src = vertices_array[i].copy()

main_pc = vertices_array[0]
main_color = colors[0]
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

    j = 1
    while j <= index:
        # apply all transformation matrices
        current_transf = transf_matrices[j]
        current_pc = current_transf @ current_pc
        j = j + 1

    # transpose current point cloud back to have info in xyz coordinates
    current_pc = current_pc.T
    current_pc = np.delete(current_pc, 3, axis=1)

    # append current point cloud to the main point cloud
    main_pc = np.vstack((main_pc, current_pc))

    # append the color as well
    current_color = colors[index]
    main_color = np.vstack((main_color, current_color))

    # move to the next point cloud
    index = index + 1

export_numpy_array_to_ply(main_pc, main_color, "result.ply")
