
import numpy as np
import pyrealsense2 as rs

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

# Start streaming
pipeline.start(config)

while True:
    frames = pipeline.wait_for_frames()
    for frame in frames:
        motion_frame = frame.as_motion_frame()
        print(motion_frame.get_profile().stream_type())
        print(motion_frame.get_profile().format())
        print(motion_frame.get_timestamp())
        motion_data = motion_frame.get_motion_data()
        # prints: x: -0.0294199, y: -7.21769, z: -6.41355 for me
        # to get numpy array:
        print(np.array([motion_data.x, motion_data.y, motion_data.z]))

