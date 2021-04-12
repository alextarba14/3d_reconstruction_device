import numpy as np


class KalmanFilter:
    """
        Compute the Kalman filter over the given measurement
        Args:
            measurement: a 1 dimensional numpy array or a list
        Returns:
            Filtered data
        ---------------------------------------------------------------
        Predict next state:
            updated[i] = F* updated[i-1] + B * u
        Predict next covariance:
            P[i] = F*P[i-1]*F_T + Q
        Compute the Kalman gain:
            K = P[i] * H_T/(H*P[i]*H_T + R)
        Update the state estimate:
            updated[i] = updated[i] + K*(measurement[i]-H*updated[i])
        Update covariance estimation:
            P[i] = (I-K*H)*P[i]
        -----------------------------------------------------------------
        F=1;
        B=0;    no control input
        H=1;    only one observable
        Q=1e-9; process noise covariance
        R;      observation noise covariance
        I=1;    identity
        -----------------------------------------------------------------
        updated[i] = updated[i-1]
        P[i] = P[i-1] + Q
        K = P[i]/(P[i] + R)
        updated[i] = updated[i] + K*(measurement[i]-updated[i])
        P[i] = (1-K)*P[i]
        -----------------------------------------------------------------
        K = P[i]/(P[i] + R)
        updated[i] = updated[i-1] + K*(measurement[i]-updated[i-1])
        P[i] = (1-K)*P[i] + Q
    """

    def __init__(self, ):
        self.P = 1
        self.Q = 1e-10
        self.R = 1e-7
        self.last_value = 0

    def filter_data(self, measurement):
        length = measurement.size
        updated = np.zeros(length)
        previous = self.last_value

        for i in range(0, length):
            K = self.P / (self.P + self.R)
            updated[i] = previous + K * (measurement[i] - previous)
            self.P = (1 - K) * self.P + self.Q
            previous = updated[i]

        self.last_value = previous
        return updated
