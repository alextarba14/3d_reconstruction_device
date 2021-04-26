"""
Usage:
------
Mouse:
    Drag with left button to rotate around pivot (thick small axes),
    with right button to translate and the wheel to zoom.
Keyboard:
    [p]     Pause
    [r]     Reset View
    [d]     Cycle through decimation values
    [z]     Toggle point scaling
    [c]     Toggle color source
    [s]     Save PNG (./out.png)
    [e]     Export points to ply (./out.ply)
    [q\ESC] Quit
"""

import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from motion import process_gyro, process_accel
from mathematics.matrix import get_matrix_average, get_matrix_sum_by_columns, create_rotation_matrix, \
    create_transformation_matrix, \
    get_matrix_median, get_trapz_integral_by_time, get_indexes_of_valid_points, get_double_trapz_integral_by_time, \
    remove_gravity_from_accel_data
from mathematics.vector import get_difference_item, get_texture_from_pointcloud
from mathematics.transformations import apply_transformations, get_kalman_filtered_data
from mathematics.kalman_filter import KalmanFilter
from export.ply import export_numpy_array_to_ply, default_export_points
from angle import Angle


class AppState:

    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.prev_position = 0, 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1
        out.fill(0)

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)


state = AppState()

WIDTH = 1280
HEIGHT = 720
REFRESH_RATE = 15
ACCEL_RATE = 250
GYRO_RATE = 200

# Configure gyro, accel, depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# enabling depth stream
config.enable_stream(rs.stream.depth, rs.format.z16, 30)
config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Create a different pipeline for IMU
imu_pipeline = rs.pipeline()
imu_config = rs.config()
# Configuring streams at different rates
# Accelerometer available FPS: {63, 250}Hz
imu_config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, ACCEL_RATE)  # acceleration
# Gyroscope available FPS: {200,400}Hz
imu_config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, GYRO_RATE)  # gyroscope
imu_profile = imu_pipeline.start(imu_config)

# Get stream profile and camera intrinsics
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

# Processing blocks
pc = rs.pointcloud()
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
colorizer = rs.colorizer()

# used to prevent false camera rotation
first = True


def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        state.mouse_btns[0] = True

    if event == cv2.EVENT_LBUTTONUP:
        state.mouse_btns[0] = False

    if event == cv2.EVENT_RBUTTONDOWN:
        state.mouse_btns[1] = True

    if event == cv2.EVENT_RBUTTONUP:
        state.mouse_btns[1] = False

    if event == cv2.EVENT_MBUTTONDOWN:
        state.mouse_btns[2] = True

    if event == cv2.EVENT_MBUTTONUP:
        state.mouse_btns[2] = False

    if event == cv2.EVENT_MOUSEMOVE:

        h, w = out.shape[:2]
        # dx, dy din accelerometru din 2 frame-uri consecutive
        # inlocuiesc miscarea mouse-ului
        # in while True
        # dezactivare ms callback
        # frustrum | fara out.fill(0)
        dx, dy = x - state.prev_mouse[0], y - state.prev_mouse[1]

        if state.mouse_btns[0]:
            state.yaw += float(dx) / w * 2
            state.pitch -= float(dy) / h * 2

        elif state.mouse_btns[1]:
            dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
            state.translation -= np.dot(state.rotation, dp)

        elif state.mouse_btns[2]:
            dz = math.sqrt(dx ** 2 + dy ** 2) * math.copysign(0.01, -dy)
            state.translation[2] += dz
            state.distance -= dz

    if event == cv2.EVENT_MOUSEWHEEL:
        dz = math.copysign(0.1, flags)
        state.translation[2] += dz
        state.distance -= dz

    state.prev_mouse = (x, y)


cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow(state.WIN_NAME, w, h)
cv2.setMouseCallback(state.WIN_NAME, mouse_cb)


def project(v):
    """project 3d vector array to 2d"""
    h, w = out.shape[:2]
    view_aspect = float(h) / w

    # ignore divide by zero for invalid depth
    with np.errstate(divide='ignore', invalid='ignore'):
        proj = v[:, :-1] / v[:, -1, np.newaxis] * \
               (w * view_aspect, h) + (w / 2.0, h / 2.0)

    # near clipping
    znear = 0.03
    proj[v[:, 2] < znear] = np.nan
    return proj


def view(v):
    """apply view transformation on vector array"""
    return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation


def line3d(out, pt1, pt2, color=(0x80, 0x80, 0x80), thickness=1):
    """draw a 3d line from pt1 to pt2"""
    p0 = project(pt1.reshape(-1, 3))[0]
    p1 = project(pt2.reshape(-1, 3))[0]
    if np.isnan(p0).any() or np.isnan(p1).any():
        return
    p0 = tuple(p0.astype(int))
    p1 = tuple(p1.astype(int))
    rect = (0, 0, out.shape[1], out.shape[0])
    inside, p0, p1 = cv2.clipLine(rect, p0, p1)
    if inside:
        cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)
    else:
        print("Outside: (" + str(p0) + ", " + str(p1) + ")")


def point3d(out, point, color=(0x00, 0xFF, 0x00), thickness=4):
    """Draw a 3D point on the view."""
    line3d(out, point, point, color, thickness)


def grid(out, pos, rotation=np.eye(3), size=1, n=20, color=(0x80, 0x80, 0x80)):
    """Draw a grid on xyz plane"""
    pos = np.array(pos)
    s = size / float(n)
    s2 = 0.5 * size

    """draw a grid on xz plane"""
    for i in range(0, n + 1):
        x = -s2 + i * s
        line3d(out, view(pos + np.dot((x, 0, -s2), rotation)),
               view(pos + np.dot((x, 0, s2), rotation)), color)

    z = 0
    for i in range(0, n + 1):
        z = -s2 + i * s
        line3d(out, view(pos + np.dot((-s2, 0, z), rotation)),
               view(pos + np.dot((s2, 0, z), rotation)), color)

    """draw a grid on xy plane"""
    for i in range(0, n + 1):
        y = -0.5 - s2 + i * s
        line3d(out, view(pos + np.dot((-s2, y, z), rotation)),
               view(pos + np.dot((s2, y, z), rotation)), color)
    for i in range(0, n + 1):
        x = - s2 + i * s
        line3d(out, view(pos + np.dot((x, -1, z), rotation)),
               view(pos + np.dot((x, 0, z), rotation)), color)


def axes(out, pos, rotation=np.eye(3), size=0.075, thickness=2):
    """draw 3d axes"""
    line3d(out, pos, pos +
           np.dot((0, 0, size), rotation), (0xff, 0, 0), thickness)
    line3d(out, pos, pos +
           np.dot((0, size, 0), rotation), (0, 0xff, 0), thickness)
    line3d(out, pos, pos +
           np.dot((size, 0, 0), rotation), (0, 0, 0xff), thickness)


def frustum(out, intrinsics, color=(0xF0, 0xF0, 0xF0)):
    """draw camera's frustum"""
    orig = view([0, 0, 0])
    w, h = intrinsics.width, intrinsics.height

    for d in range(1, 6, 2):
        def get_point(x, y):
            p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
            line3d(out, orig, view(p), color)
            return p

        top_left = get_point(0, 0)
        top_right = get_point(w, 0)
        bottom_right = get_point(w, h)
        bottom_left = get_point(0, h)

        line3d(out, view(top_left), view(top_right), color)
        line3d(out, view(top_right), view(bottom_right), color)
        line3d(out, view(bottom_right), view(bottom_left), color)
        line3d(out, view(bottom_left), view(top_left), color)


def pointcloud(out, verts, texcoords, color, painter=True):
    """draw point cloud with optional painter's algorithm"""
    if painter:
        # Painter's algo, sort points from back to front

        # get reverse sorted indices by z (in view-space)
        # https://gist.github.com/stevenvo/e3dad127598842459b68
        v = view(verts)
        # returns indices in reverse order
        s = v[:, 2].argsort()[::-1]
        proj = project(v[s])
    else:
        proj = project(view(verts))

    if state.scale:
        proj *= 0.5 ** state.decimate

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
    np.clip(u, 0, ch - 1, out=u)
    np.clip(v, 0, cw - 1, out=v)

    # perform uv-mapping
    out[i[m], j[m]] = color[u[m], v[m]]


# initializing empty 3D array
out = np.empty((h, w, 3), dtype=np.uint8)
# grid(out, (0.2, 0, 0.2), size=1, n=30)

vertices = []
vertices_copy =[]
tex_coords = []
color_frames = []
transf_matrices = []

threshold = 20
frame_count = -1
accel_state = [0, -9.81, 0, 1]
accel_data_array = [[0 for x in range(3)] for y in range(threshold)]
gyro_data_array = [[0 for x in range(3)] for y in range(threshold)]
# instantiate two different KalmanFilter objects for each gyro and accelerometer
accel_KF_x = KalmanFilter()
accel_KF_y = KalmanFilter()
accel_KF_z = KalmanFilter()
gyro_KF_x = KalmanFilter()
gyro_KF_y = KalmanFilter()
gyro_KF_z = KalmanFilter()
index = 0
mat_count = -1
while True:
    # Grab camera data
    if not state.paused:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        # incrementing frame count
        frame_count = frame_count + 1
        theta = Angle(0, 0, 0)

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
                # print(gyro_data)

            h, w = out.shape[:2]

            # Getting first rotation information in order to prevent
            # a false rotation because the previous positions were 0,0
            # if first:
            #     first = False
            #     state.prev_position = (theta.x, theta.y, theta.z)
            # # getting movement
            # dx, dy, dz = theta.x - state.prev_position[0], theta.y - state.prev_position[1], theta.z - \
            #              state.prev_position[2]
            #
            # # updating view with new movement values
            # state.yaw += float(dy)
            # state.pitch -= float(dx)
            # state.translation[2] += dz

            # updating current position
            # state.prev_position = (theta.x, theta.y, theta.z)

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

        pc.map_to(mapped_frame)
        points = pc.calculate(depth_frame)

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
            # remove noise from gyro data using the Kalman filter
            # noiseless_gyro = get_kalman_filtered_data(gyro_data_array, gyro_KF_x, gyro_KF_y, gyro_KF_z)
            # remove the gravity from the acceleration samples using noiseless gyro data
            acceleration = remove_gravity_from_accel_data(accel_data_array, gyro_data_array, accel_state)
            # remove noise from acceleration data using the Kalman filter
            # noiseless_acceleration = get_kalman_filtered_data(acceleration, accel_KF_x, accel_KF_y, accel_KF_z)
            # noiseless_acceleration = acceleration
            translation = get_double_trapz_integral_by_time(acceleration)

            # update the accelerometer state with the last data
            accel_state = accel_data_array[9]

            # get the rotation angle by integrating the rotation velocity
            rotation = get_trapz_integral_by_time(gyro_data_array)
            rotation_matrix = create_rotation_matrix(rotation)

            # multiply rotation matrix with translation matrix in homogeneous coordinates
            transf_mat = create_transformation_matrix(rotation_matrix, translation)
            # points.export_to_ply(f'./out{mat_count}.ply', mapped_frame)
            file_name = f'original_out{mat_count}.ply'
            # default_export_points(points, file_name)

            # append to bigger lists
            # get the valid point indexes from the vertexes array
            valid_points_indexes = get_indexes_of_valid_points(verts)
            vertices.append(verts[valid_points_indexes])
            vertices_copy.append(verts[valid_points_indexes])
            transf_matrices.append(transf_mat)
            color_frames.append(color_frame)
            # use the same indexes as the vertices
            tex_coords.append(texcoords[valid_points_indexes])

            mat_count = mat_count + 1

        if mat_count == 8:
            updated_pointclouds = apply_transformations(vertices, transf_matrices)
            for index in range(len(updated_pointclouds)):
                texture = get_texture_from_pointcloud(updated_pointclouds[index], tex_coords[index],
                                                      color_frames[index])
                # save the transformed pointcloud
                file_name = f'./test/transformed_using_trapz{index}.ply'
                export_numpy_array_to_ply(updated_pointclouds[index], texture, file_name=file_name)
                # file_name = f'./test/original{index}.ply'
                # export_numpy_array_to_ply(vertices_copy[index], texture, file_name=file_name)
            exit(1)

    # Render
    now = time.time()

    # refreshing output
    # out.fill(0)

    orig = view([0, 0, 0])
    point3d(out, orig)
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

    # if any(state.mouse_btns):
    #     axes(out, view(state.pivot), state.rotation, thickness=4)

    dt = time.time() - now

    cv2.setWindowTitle(
        state.WIN_NAME, "RealSense (%dx%d) %dFPS (%.2fms) %s" %
                        (w, h, 1.0 / dt, dt * 1000, "PAUSED" if state.paused else ""))

    cv2.imshow(state.WIN_NAME, out)
    key = cv2.waitKey(1)

    if key == ord("r"):
        # Refresh view
        state = AppState()
        out.fill(0)
        grid(out, (0.2, 0, 0.2), size=1, n=30)

    if key == ord("p"):
        state.paused ^= True

    if key == ord("d"):
        state.decimate = (state.decimate + 1) % 3
        decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)

    if key == ord("z"):
        state.scale ^= True

    if key == ord("c"):
        state.color ^= True

    if key == ord("s"):
        cv2.imwrite('./out.png', out)

    if key == ord("e"):
        default_export_points(points)

    if key in (27, ord("q")) or cv2.getWindowProperty(state.WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
        break

# Stop streaming
pipeline.stop()
