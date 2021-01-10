from angle import Angle
import numpy as np
import pyrealsense2 as rs
import math

def process_accel(accel_data):
    """
    Computes the rotation angle from accelerometer data and updates the current theta.
    accel_data is an rs2_vector that holds the measurements retrieved from the accelerometer stream.
    """
    # Holds the angle as calculated from accelerometer data
    # with x,y,z coordinates
    accel_angle = Angle(0,0,0)

    # Calculate rotation angle from accelerometer data
    # accel_angle.z = atan2(accel_data.y, accel_data.z);
    accel_angle.z = math.atan2(accel_data.y,accel_data.z)

    # accel_angle.x = atan2(accel_data.x, sqrt(accel_data.y * accel_data.y + accel_data.z * accel_data.z));
    accel_angle.x = math.atan2(accel_data.x,math.sqrt(accel_data.y * accel_data.y + accel_data.z * accel_data.z))

    global first
    global theta
    global alpha

    # If it is the first iteration set initial pose of camera according to accelerometer data
    if first:
        first = False
        theta = accel_angle
        theta.y = math.pi
    else:
        """
        Apply Complementary Filter:
        - 'high-pass filter' = theta * alpha: allows short duration signals to pass
        through while filtering out signals that are steady over time, is used to cancel out drift.
        - 'low-pass filter' = accel * (1-alpha): lets through long term changes,
        filtering out short term fluctuations
        - alpha is used to aggregate the data
        """
        theta.x = theta.x * alpha + accel_angle.x * (1-alpha)
        theta.z = theta.z * alpha + accel_angle.z * (1-alpha)

# Global variables
alpha = 0.98
first = True
theta = Angle(0, 0, 0)

def main():
    # Configure gyro and accelerometer streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
    config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

    # Start streaming
    pipeline.start(config)

    while True:
        frames = pipeline.wait_for_frames()
        for frame in frames:
            motion = frame.as_motion_frame()
            if motion and motion.get_profile().stream_type() == rs.stream.accel:
                # Accelerometer frame
                # Get the timestamp of current frame
                timestamp = motion.get_timestamp()
                # Get accelerometer measurements
                accel_data = motion.get_motion_data()
                process_accel(accel_data)

            motion_data = motion.get_motion_data()
            # prints: x: -0.0294199, y: -7.21769, z: -6.41355 for me
            # to get numpy array:
            print(np.array([motion_data.x, motion_data.y, motion_data.z]))


if __name__ == "__main__":
    main()
