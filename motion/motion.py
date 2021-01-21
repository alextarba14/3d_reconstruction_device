import threading
from angle import Angle
import numpy as np
import pyrealsense2 as rs
import math
from timeit import default_timer as timer

# Global variables
alpha = 0.98
first = True
theta = Angle(0, 0, 0)
last_timestamp_gyro = 0

# Mutex primitive
mutex = threading.Lock()

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
    global mutex

    mutex.acquire()
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
    mutex.release()
    print("Theta angle(accel): " + str(theta))


def process_gyro(gyro_data, timestamp):
    """
    Computes the change in rotation angle, based on gyroscope measurements.
    It accepts gyro_data, an rs2_vector containing measurements retrieved from gyroscope,
    and timestamp, the timestamp of the current frame from gyroscope stream.
    """
    global first
    global last_timestamp_gyro

    # On the first iteration use only data from accelerometer
    # to set the camera's initial position
    if first:
        last_timestamp_gyro = timestamp
        return

    # Initialize gyro angle with data from gyro
    # gyro_data.x : Pitch
    # gyro_data.y : Yaw
    # gyro_data.z : Roll
    gyro_angle = Angle(gyro_data.x,gyro_data.y,gyro_data.z)
    print("Gyro angle (before): " + str(gyro_angle))
    # Compute the difference between arrival times of previous and current gyro frames
    dt_gyro = (timestamp - last_timestamp_gyro) / 1000.0
    last_timestamp_gyro = timestamp
    print("dt_gyro: " + str(dt_gyro))
    # Change in angle equals gyro measurements * time passed since last measurement
    gyro_angle = gyro_angle * dt_gyro

    print("Gyro angle(after): " + str(gyro_angle))

    # Apply the calculated change of angle to the current angle (theta)
    global mutex
    mutex.acquire()
    theta.add(-gyro_angle.z,-gyro_angle.y,gyro_angle.x)
    mutex.release()
    print("Theta angle(gyro): " + str(theta))



def main():
    start = timer()
    nr = 0
    # Configure gyro and accelerometer streams
    pipeline = rs.pipeline()
    config = rs.config()
    # Configuring streams at different rates
    # Accelerometer available FPS: {63, 250}Hz
    # Gyroscope available FPS: {200,400}Hz
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
                # Get accelerometer measurements
                accel_data = motion.get_motion_data()
                print("Accel data: " + str(accel_data))
                process_accel(accel_data)
            elif motion and motion.get_profile().stream_type() == rs.stream.gyro:
                # Gyro frame
                # Get the timestamp of current frame
                timestamp = motion.get_timestamp()
                # Get gyro measurements
                gyro_data = motion.get_motion_data()
                print("gyro_data : " + str(gyro_data))
                process_gyro(gyro_data, timestamp)
        nr = nr + 1
        end = timer()
        if end - start >= 5:
            print("Stopped! Nr: " + str(nr))
            break




if __name__ == "__main__":
    main()
