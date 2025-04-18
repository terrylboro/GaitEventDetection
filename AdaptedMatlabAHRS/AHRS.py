# A class which implements the Matlab AHRS filter algorithm in Python
# https://uk.mathworks.com/help/fusion/ref/ahrsfilter-system-object.html
# Written by Terry Fawden 16/8/23

# import functions from the submodules
# myDir = os.getcwd()
# sys.path.append(myDir)
#
# from pathlib import Path
# path = Path(myDir)
# a=str(path.parent.absolute())
#
# sys.path.append(a)
import os
print(os.getcwd())
import sys
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
from AdaptedMatlabAHRS.utils.ahrs_plotting_utils import plot_imu_xyz, plot_euler_angles
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from itertools import compress
from AdaptedMatlabAHRS.utils.quaternion_utils import *


class AHRS:
    """ A Python implementation of the Matlab AHRS filter """
    def __init__(self, is6axis : bool):
        """ Setup the default properties of the filter and input arrays
        \nInputs:
        accelReadings -- np.array(Nx3) float in m/s^2\n
        gyroReadings -- np.array(Nx3) float in rad/s\n
        magReadings -- np.array(Nx3) float in uT """
        # set the default properties of the filter
        self.SampleRate = 100  # Adapt according to collection protocol
        self.is6axis = is6axis  # Exclude/include magnetometer
        self.ExpectedMagneticFieldStrength = 49.2 # Expected estimate of magnetic field strength (μT)
        # set noise properties
        self.AccelerometerNoise = 0.00019247  # Variance of accelerometer signal noise ((m/s2)2)
        self.MagnetometerNoise = 0.1  # Variance of magnetometer signal noise (μT2)
        self.GyroscopeNoise = 9.1385 * 10 ** -3  # 9.1385 * 10**-5 # Variance of gyroscope signal noise ((rad/s)2)
        self.GyroscopeDriftNoise = 3.0462 * 10 ** -13  # Variance of gyroscope offset drift ((rad/s)2)
        self.LinearAccelerationNoise = 3.2#3.2#0.85#0.0096236  # Variance of linear acceleration noise (m/s2)2
        self.LinearAccelerationDecayFactor = 0.01  # 0.5 # Decay factor for linear drift, range [0,1]. Set low if linear accel. changes quickly
        self.MagneticDisturbanceNoise = 0.25  # Variance of magnetic disturbance noise (μT2)
        self.MagneticDisturbanceDecayFactor = 0.1  # Decay factor for magnetic disturbance

        # initialise internal state variables
        self.isFirst = True
        self.linAccelPrior = np.zeros(3)  # assume stationary start
        self.gyroOffset = np.zeros(3)  # set this according to pre calibration
        self.m = self.ExpectedMagneticFieldStrength * np.array([1, 0, 0])
        self.Qw = np.diag(np.array([0.000006092348395734171, 0.000006092348395734171, 0.000006092348395734171,
                                0.00007615435494667714, 0.00007615435494667714, 0.00007615435494667714,
                                0.009623610000000, 0.009623610000000, 0.009623610000000,
                                0.600000000000000, 0.600000000000000, 0.600000000000000]))
        # Can precalculate observation model noise, Qv, since this doesn't change
        self._calculate_observation_model_noise()

    def run(self, imuMeasurements, plot=False):
        """
        Apply the particle filtering algorithm to the preprocessed NED c3dData.
        Note that q_plus is initialised using eCompass during first pass.
        linAccelPrior has to be initialised as zero - impose 'standing still' start condition. Not mentioned in documentation.
        initial gyro offset can be determined beforehand using calibration. It should adapt throughout anyway.
        :param plot: Whether to plot the orientation output.
        :type plot: bool
        :param is6axis: Select whether using accelerometer and gyroscope alone (True) or including magnetometer (False)
        :type is6axis: bool
        """
        # Arrange the input and intialisation variables
        # assign the IMU c3dData
        accelReadings = imuMeasurements[:, 0:3]
        gyroReadings = imuMeasurements[:, 3:6]
        if not self.is6axis:
            magReadings = imuMeasurements[:, 6:9]
        else:
            magReadings = np.zeros_like(accelReadings)
            magReadings[:, 0] = 1
        # Set initial rotation estimate according to eCompass
        self.q_plus, _ = self.eCompass(accelReadings[0, :], magReadings[0, :])
        # setup output variables
        M = np.size(accelReadings, 0)
        self.orientation = np.zeros((M, 4))
        self.orientation_euler = np.zeros((M, 3))
        g_array = np.zeros((M, 3))
        self.angularVelocity = np.zeros((M, 3))
        self.rotmat = np.zeros((M, 3, 3))
        self.linAccArr = np.zeros((M, 3))
        # Set noise covariance matrix
        self.Qw = self.Qw[0:9, 0:9] if self.is6axis else self.Qw
        for i in range(0, M):
            # Calculate offset-corrected angular velocity
            self.angularVelocity[i] = gyroReadings[i, :] - self.gyroOffset
            # Execute process model prediction
            self.update_measurement_model(accelReadings[i, :], gyroReadings[i, :])
            # Update the error state vector
            self.update_error_state(magReadings[i, :], self.is6axis)
            # Perform prediction and update steps using Kalman equations
            self.kalman_forward()
            # Perform magnetometer correction
            if not self.is6axis:
                self.mError = np.matmul(self.K[9:12], self.z.T).T
            # Perform correction step
            self.correct()
            # Update magnetic vector using corrected a posteriori information
            if not self.is6axis:
                self.update_magnetic_vector()
            # Append orientation quaternion to list
            # self._debug_log(i+1)
            self.orientation[i] = self.q_plus
            self.orientation_euler[i] = R.from_quat(self.q_plus).as_euler('zyx', degrees=True)
            self.rotmat[i] = R.from_quat(self.q_plus).as_matrix()
            self.linAccArr[i] = self.linAccelPost
        # plot findings
        if plot:
            # plot_imu_xyz(accelReadings, gyroReadings, magReadings, range(0, len(accelReadings)),
            #              "NED IMU Data", 1)
            # plt.show()
            plot_euler_angles(range(0, len(accelReadings)), self.orientation_euler,
                              "Euler Angles", 2)
            plt.show()

    def update_measurement_model(self, accelReading : np.array, gyroReading : np.array):
        """
            Execute the Model section of the AHRS algorithm. This estimates acceleration due to gravity in the accelerometer
            and gyroscope and the magnetic field and produces a prior estimate of orientation.
            :param imuMeasurements: XYZ IMU readings (in order accelerometer, gyroscope, magnetometer)
            :param gyroOffset: XYZ gyroscope offset
            :param linAccelPrior: XYZ estimate of linear acceleration
            :param m: XYZ estimate of magnetic field
            :param q_plus: Orientation estimate quaternion
            :param iteration: Number of iterations the filter has completed so far
            :param is6axis: Toggle magnetometer calculations on/off, default is off
            :return: Gravitational vector estimate from accelerometer, magnetic field estimate from gyroscope and magnetometer,
            gravitational vector estimate from gyroscope, prior estimate of orientation, prior estimate of linear acceleration
            """
        # Ensure that q_plus has been initialised before first iteration using eCompass
        # Calculate a priori orientation estimate using gyroscope
        delta_psi = (gyroReading - self.gyroOffset) / self.SampleRate
        deltaQ = R.from_rotvec(delta_psi).as_quat(canonical=True)
        self.q_minus = quaternion_multiply(self.q_plus, deltaQ)
        # Estimate gravitational and magnetic vectors from new orientation
        rPrior = quaternion_rotation_matrix(self.q_minus)
        self.g = rPrior[:, 2] * 9.81
        self.mGyro = np.matmul(rPrior, self.m.T).T.flatten()
        # Estimate gravity from accelerometer according to
        self.pLinAccelPrior = self.LinearAccelerationDecayFactor * self.linAccelPrior
        self.gAccel = accelReading + self.pLinAccelPrior


    def update_error_state(self, magReading : np.array, is6Axis):
        """
        Calculate the error between accelerometer and gyroscope estimates of gravity
        and between gyroscope estimate and magnetometer-derived magnetic vector.
        :param magReading: 1 x 3 numpy array - latest reading from magnetometer.
        :param is6Axis: Determines whether to include magnetic vector calulation.
        """
        # Diff between the gravity estimate from the accelerometer readings and the gyroscope readings
        self.z_g = self.gAccel - self.g
        if not self.is6axis:
            # Diff between magnetic vector estimate from the gyroscope readings and magnetometer
            self.z_m = magReading - self.mGyro
            # z is simply a concatenation of the two
            self.z = np.concatenate((self.z_g, self.z_m), axis=0)
        else:
            self.z = self.z_g


    def update_magnetic_vector(self):
        """
        Update magnetic vector estimate, old_m using the a posteriori magnetic disturbance error
        and the a posteriori orientation.
        """
        rPost = quaternion_rotation_matrix(self.q_plus)
        mErrorNED = np.matmul(rPost.T, self.mError.T).T
        self.m = np.reshape(self.m, 3)
        # The magnetic disturbance error in the navigation frame is subtracted from the previous magnetic vector estimate and then interpreted as inclination:
        M = self.m - mErrorNED
        inclination = np.arctan2(M[2], M[0])
        # The inclination is converted to a constrained magnetic vector estimate for the next iteration:
        self.m = np.zeros((1, 3))
        self.m[0, 0] = self.ExpectedMagneticFieldStrength * np.cos(inclination)
        self.m[0, 2] = self.ExpectedMagneticFieldStrength * np.sin(inclination)


    def kalman_forward(self):
        if self.is6axis:
            self.H = self._compute_measurement_matrix_6axis()  # Compute the observation model (H) matrix
        else:
            self.H = self._compute_measurement_matrix_9axis()  # Compute the observation model (H) matrix
        # innovation covariance, S aka tmp
        self.S = (self.H @ self.Qw @ self.H.T + self.Qv).T
        # Calculate the Kalman gain, K_12x6
        # self.K = np.matmul(np.matmul(self.Qw, self.H.T), np.linalg.inv(self.S))
        self.K = self.Qw @ self.H.T @ np.linalg.inv(self.S)
        self.x = self.K @ self.z
        # Update error covariance estimate - is this equation correct?
        self.P = self.Qw - (self.K @ self.H @ self.Qw)
        self._predict_error_estimate_covariance()


    def correct(self):
        """
        Execute the Correct section of the MATLAB AHRS algorithm.
        """
        theta_plus = np.reshape(self.x[0:3], 3)
        a_plus = np.reshape(self.x[6:9], 3)
        b_plus = np.reshape(self.x[3:6], 3)
        # Multiply prior orientation estimate by previously calculated orientation error
        theta_plus = quaternion_conj(R.from_rotvec(theta_plus).as_quat(canonical=True))
        self.q_plus = quaternion_multiply(self.q_minus, theta_plus)
        self.q_plus = quaternion_normalise(self.q_plus)
        # Correct linear acceleration and gyroscope bias using error
        self.linAccelPost = self.pLinAccelPrior - a_plus
        self.gyroOffset = self.gyroOffset - b_plus

    @staticmethod
    def eCompass(accel : np.array, mag : np.array):
        """
        Compute the 3D orientation of a device from its accelerometer and magnetometer c3dData.
        Can be used to initialise the filter
        :param accel: 1x3 numpy array containing XYZ accelerometer c3dData
        :param mag: 1x3 numpy array containing XYZ magnetometer c3dData
        :return: orientation in quaternion and rotation matrix formats
        """
        down = accel / np.linalg.norm(accel)
        east = np.cross(down, mag)
        east /= np.linalg.norm(east)
        north = np.cross(east, down)
        rot_matrix = np.vstack([north, east, down])
        rotation = R.from_matrix(rot_matrix)
        euler_angles = rotation.as_euler('zyx', degrees=True)
        q = R.from_matrix(rot_matrix).as_quat(canonical=True)
        return q, euler_angles

    def _calculate_observation_model_noise(self):
        accel_noise = self.AccelerometerNoise + self.LinearAccelerationNoise + (1 / self.SampleRate) ** 2 * (self.GyroscopeDriftNoise + self.GyroscopeNoise)
        mag_noise = self.MagnetometerNoise + self.MagneticDisturbanceNoise + (1 / self.SampleRate) ** 2 * (self.GyroscopeDriftNoise + self.GyroscopeNoise)
        if self.is6axis:
            self.Qv = np.diag(np.array([accel_noise, accel_noise, accel_noise]))
        else:
            self.Qv = np.diag(np.array([accel_noise, accel_noise, accel_noise, mag_noise, mag_noise, mag_noise]))

    def _predict_error_estimate_covariance(self):
        """
        Predict the error estimate covariance (Qw) according to P
        """
        beta = self.GyroscopeDriftNoise
        epsilon = self.GyroscopeNoise
        nu = self.LinearAccelerationDecayFactor
        xi = self.LinearAccelerationNoise
        gamma = self.MagneticDisturbanceNoise
        sigma = self.MagneticDisturbanceDecayFactor
        k = (1 / self.SampleRate)
        self.Qw[0, 0] = self.P[0, 0] + k ** 2 * (self.P[3, 3] + beta + epsilon)
        self.Qw[3, 0] = -k * (self.P[3, 3] + beta)
        self.Qw[1, 1] = self.P[1, 1] + k ** 2 * (self.P[4, 4] + beta + epsilon)
        self.Qw[4, 1] = -k * (self.P[4, 4] + beta)
        self.Qw[2, 2] = self.P[2, 2] + k ** 2 * (self.P[5, 5] + beta + epsilon)
        self.Qw[5, 2] = -k * (self.P[5, 5] + beta)
        self.Qw[0, 3] = -k * (self.P[3, 3] + beta)
        self.Qw[3, 3] = self.P[3, 3] + beta
        self.Qw[1, 4] = -k * (self.P[4, 4] + beta)
        self.Qw[4, 4] = self.P[4, 4] + beta
        self.Qw[2, 5] = -k * (self.P[5, 5] + beta)
        self.Qw[5, 5] = self.P[5, 5] + beta
        # onto the accelerometer section
        self.Qw[6, 6] = nu ** 2 * self.P[6, 6] + xi
        self.Qw[7, 7] = nu ** 2 * self.P[7, 7] + xi
        self.Qw[8, 8] = nu ** 2 * self.P[8, 8] + xi
        if not self.is6axis:
            self.Qw[9, 9] = sigma ** 2 * self.P[9, 9] + gamma
            self.Qw[10, 10] = sigma ** 2 * self.P[10, 10] + gamma
            self.Qw[11, 11] = sigma ** 2 * self.P[11, 11] + gamma

    @staticmethod
    def _skew_matrix(v : np.array):
        """ Make a skew-symmetrix matrix from a 1x3 vector """
        h = np.zeros((3, 3))
        h[0, 1] = v[2]
        h[0, 2] = -v[1]
        h[1, 2] = v[0]
        return h - h.T

    def _compute_measurement_matrix_9axis(self):
        """ Compute the measurement matrix, H """
        h1 = self._skew_matrix(self.g)
        h2 = self._skew_matrix(self.mGyro)
        h3 = -h1 * (1/self.SampleRate)
        h4 = -h2 * (1/self.SampleRate)
        top_H = np.concatenate((np.array(h1), np.array(h3), np.identity(3), np.zeros((3, 3))), axis=1)
        bottom_H = np.concatenate((np.array(h2), np.array(h4), np.zeros((3, 3)), -1 * np.identity(3)), axis=1)
        H = np.vstack((top_H, bottom_H))
        return np.reshape(H, (6, 12))

    def _compute_measurement_matrix_6axis(self):
        """ Compute the measurement matrix, H """
        h1 = self._skew_matrix(self.g)
        h3 = -h1 * (1/self.SampleRate)
        return np.concatenate((np.array(h1), np.array(h3), np.identity(3)), axis=1)

    def _debug_log(self, iteration):
        """
        Print internal information after each loop.
        """
        print("--- Summary after step {} ---".format(iteration))
        # print("--- H ----")
        # print(self.H)
        # print("--- q+ ----")
        # print(self.q_plus)
        # print("--- pQw ----")
        # print(self.Qw)
        # print("--- linAccelPost ----")
        # print(self.linAccelPost)
        # print("--- gyro offset ----")
        # print(self.gyroOffset)
        # print("mGyro: ", self.mGyro)
        # print("z: ", self.z)
        # print("S: ", self.S)
        # print("K: ", self.K)
        # print("mError: ", self.mError)
        # print("m: ", self.m)
        print("Qw: ", self.Qw)



def test():
    # firstly load the IMU df
    # load real accelerometer c3dData
    df = pd.read_csv('../data/TF_64-31.csv')
    magMeasurements = np.zeros((len(df), 3))
    # N: 19,177.3 nT	E: 244.9 nT	D: 45,347.7 nT	Total: 49,236.6 nT
    earthMagVec = np.array([19.177, 0.2449, 10])
    df[['MagXrear', 'MagYrear', 'MagZrear']] += (earthMagVec - df.loc[0, ['MagXrear', 'MagYrear', 'MagZrear']])  #6.5, 14, 77.5
    magMeasurements = df[['MagXrear', 'MagYrear', 'MagZrear']].astype(float).to_numpy()
    imuMeasurements = df[["AccXrear", "AccYrear", "AccZrear",
                          "GyroXrear", "GyroYrear", "GyroZrear"]].to_numpy()
    imuMeasurements = np.concatenate((imuMeasurements, magMeasurements), axis=1)

    is6axis=False
    ahrs = AHRS(is6axis=is6axis)
    ahrs.run(imuMeasurements=imuMeasurements, plot=True)

    # check linear acceleration
    plt.plot(ahrs.linAccArr)
    plt.legend(["X", "Y", "Z"])
    plt.title("Estimated Linear Acceleration")
    plt.show()



if __name__ == "__main__":
    test()


