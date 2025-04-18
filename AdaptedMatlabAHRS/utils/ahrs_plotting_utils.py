# Contains useful functions for plotting common c3dData use cases
# Relies on Matplotlib
# Written by Terry Fawden 22/8/2023

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_imu_xyz(accel, gyro, mag, time, title, figNum):

    fig, axs = plt.subplots(3, sharex=True, layout='constrained')

    # plt.subplot(3, 1, 1)
    axs[0].plot(time, accel[:, 0], 'b-')
    axs[0].plot(time, accel[:, 1], 'r-')
    axs[0].plot(time, accel[:, 2], 'y-')
    axs[0].set(ylabel='Acceleration / m/s^2')

    # plt.subplot(3, 1, 2)
    axs[1].plot(time, gyro[:, 0], 'b-')
    axs[1].plot(time, gyro[:, 1], 'r-')
    axs[1].plot(time, gyro[:, 2], 'y-')
    axs[1].set(ylabel='Angular\n Velocity / m/s')

    # plt.subplot(3, 1, 3)
    axs[2].plot(time, mag[:, 0], 'b-')
    axs[2].plot(time, mag[:, 1], 'r-')
    axs[2].plot(time, mag[:, 2], 'y-')
    axs[2].set(xlabel='Time / samples', ylabel='Magnetic Field\n Strength / uT')

    # Hide x labels and tick labels for all but bottom plot.
    for ax in axs:
        ax.label_outer()

    fig.legend(['X', 'Y', 'Z'], loc='outside upper right')
    fig.suptitle(title, fontsize=18)

    # plt.tight_layout()

    # f.show()


def plot_accel(time, accel, title, figNum=None, c=None, legend=None):
    if figNum:
        plt.figure(figNum)
        plt.clf()
    if c:
        plot_colours = c
    else:
        plot_colours = ['b-', 'r-', 'y-']
    plt.plot(time, accel[:, 0], plot_colours[0])
    plt.plot(time, accel[:, 1], plot_colours[1])
    plt.plot(time, accel[:, 2], plot_colours[2])
    plt.title(title, fontsize=36, wrap=True)
    plt.xlabel("Time / s", fontsize=24)
    plt.ylabel("Euler Angle / deg", fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    if not legend:
        plt.legend(['X (forward)', 'Y (side-to-side)', 'Z (vertical)'])
    else:
        plt.legend(legend, fontsize=20)


def plot_euler_angles(time, euler_angle_array, title, figNum=None, c=None, legend=None):
    if figNum:
        plt.figure(figNum)
    if c:
        plot_colours = c
    else:
        plot_colours = ['b-', 'r-', 'y-']
    plt.plot(time, euler_angle_array[:, 0], plot_colours[0])
    plt.plot(time, euler_angle_array[:, 1], plot_colours[1])
    plt.plot(time, euler_angle_array[:, 2], plot_colours[2])
    plt.title(title, fontsize=36)
    plt.xlabel("Time / s", fontsize=24)
    plt.ylabel("Euler Angle / deg", fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    if not legend:
        plt.legend(['ψ - Yaw (z)', 'θ - Pitch (y)', 'φ - Roll (x)'])
    else:
        plt.legend(legend, fontsize=20)

    # f2.show()


def main():
    data = pd.read_csv('/Data/Gimbal/gimbal-routine-try3.csv')
    # c3dData = pd.read_csv('C:/Users/teri-/PycharmProjects/headinspace/gimbal-calibration.csv')
    accel = data[['accel_x', 'accel_y', 'accel_z']].values
    gyro = data[['gyro_x', 'gyro_y', 'gyro_z']].values
    mag = data[['mag_x', 'mag_y', 'mag_z']].values
    N = np.size(accel, 0)
    # Rearrange the c3dData to fit the correct format
    accelReadings = np.reshape(accel[:, :], (N, 3))
    gyroReadings = np.reshape(gyro[:, :], (N, 3))
    magReadings = np.reshape(mag[:, :], (N, 3))

    temp_accelReadings2 = accelReadings[:, 2]
    accelReadings[:, [1, 2]] = accelReadings[:, [2, 1]]
    accelReadings[:, 2] = - accelReadings[:, 2]
    gyroReadings[:, [1, 2]] = gyroReadings[:, [2, 1]]
    gyroReadings[:, 2] = - gyroReadings[:, 2]
    magReadings[:, [1, 2]] = magReadings[:, [2, 1]]
    magReadings[:, 2] = - magReadings[:, 2]

    time = range(0, len(data))

    plot_imu_xyz(accelReadings, gyroReadings, magReadings, time, "Calibration IMU Data", 1)

    # # test out plotting euler angles
    # eulerData = pd.read_csv('../../Visualisation/OrientationData/python-his-euler.csv')
    # euler_angle_array = eulerData.values
    # plot_euler_angles(time[:-1], np.rad2deg(euler_angle_array), "Euler Angles (Python)", 2)
    #
    # # compare to matlab-generated angles
    # matlabData = pd.read_csv('C:/Users/teri-/PycharmProjects/headinspace/gimbal-calibration.csv')
    # matlab_euler_angle_array = matlabData.values
    # # plot_euler_angles(time[:-1], matlab_euler_angle_array, "Euler Angles (Matlab)", 3)
    # plot_euler_angles(time, matlab_euler_angle_array, "Euler Angles (Matlab)", 3)

    plt.show()


if __name__ == "__main__":
    # execute main
    main()


