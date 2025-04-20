import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from AdaptedMatlabAHRS.AHRS import AHRS
from utils.data_manipulation_utils import calculate_acc_zero, lp_filter, first_nonzero
from utils.orientation_utils import calculate_tilt_corrected_linear_acceleration_adapted
import matplotlib as mpl
mpl.use("Qt5Agg")

if __name__ == "__main__":
    # import the data
    opticalFiledir = 'C:/Users/teri-/Documents/UnlabelledNonWalkTrials/TUGC3Ds/'
    tugEventsDF = pd.read_csv("../utils/info/TUG Gait Events.csv", index_col="Filename")
    infoDF = pd.read_csv("../utils/info/Sit2WalkInfo.csv", index_col="Filename")
    # add file to store turning events
    turningEventsDF = pd.read_csv("../utils/info/TUG Turning Events from IMU.csv", index_col="Filename")

    filelist = tugEventsDF.index.tolist()
    for opticalFilename in filelist:
        if infoDF.loc[opticalFilename, "IMUDataGood"] == "Good":
            # Retrieve files for selected activity and participant
            subjectNum = int(opticalFilename[3:5])
            trialNum = int(opticalFilename[8:10])
            if subjectNum in [1, 6]:
                trialNum += 1
            print(opticalFilename)
            if subjectNum==21 and trialNum==30: #72 > subjectNum > 0 and subjectNum not in [46, 47, 48]:
                imuFilename = "TF_{:02d}-{:02d}.csv".format(subjectNum, trialNum)
                offset = int(infoDF.at[opticalFilename, "Offset"])
                firstStepIdx = infoDF.at[opticalFilename, "FirstStepIdx"]
                turnsIdxs = tugEventsDF.loc[
                    opticalFilename, ["StartTurnIdx", "EndTurnIdx", "EndWalkIdx"]]  # + firstStepIdx + offset
                # Extract the data
                imuDF = pd.read_csv(os.path.join("../imu/data/TF_{:02d}".format(subjectNum), imuFilename))
                # Calculate orientations
                # for side, sideName in zip(["lear", "rear", "chest"], ["Left Ear", "Right Ear", "Chest"]):
                for side, sideName in zip(["lear"], ["Left Ear"]):
                    ahrs = AHRS(
                        imuDF.loc[:, ["AccX" + side, "AccY" + side, "AccZ" + side]].to_numpy(),
                        imuDF.loc[:, ["GyroX" + side, "GyroY" + side, "GyroZ" + side]].to_numpy(),
                        imuDF.loc[:, ["MagX" + side, "MagY" + side, "MagZ" + side]].to_numpy()
                    )
                    ahrs.run(plot=False, is6axis=True)

                    # Plot just the yaw
                    # This code generates the demo plot
                    import seaborn as sns
                    yaw = np.unwrap(ahrs.orientation_euler[offset:950, 0], 180)
                    x = np.linspace(0, 0.01 * (len(ahrs.orientation_euler) - offset), len(ahrs.orientation_euler) - offset)
                    x = x[:950-offset]
                    sns.lineplot(x=x, y=lp_filter(yaw, 5), linewidth=3, label="__no_legend__")
                    # plt.fill_betweenx(yaw, int(turnsIdxs[0]), int(turnsIdxs[1]), facecolor='green', alpha=.5,
                    #                     label="Turn")
                    turnsDF = pd.read_csv("../utils/info/TUG Turning Events from IMU.csv", index_col="Filename")
                    turnsIdxs = turnsDF.loc[opticalFilename, ["StartTurnIdx","EndTurnIdx"]].mul(0.01).to_numpy().tolist()
                    plt.vlines(turnsIdxs, min(yaw)-5, max(yaw)+5, color="g", linewidth=2, label="__no_legend__")
                    plt.fill_between(turnsIdxs, y1=min(yaw)-5, y2=max(yaw)+5, color="g", alpha=0.2, label="Turn")
                    plt.hlines([15, np.max(np.abs(yaw[0:int(0.85*len(yaw))])) - 5], 0, 0.01*len(yaw), color="r", linestyle="--",
                               linewidth=2, label="Thresholds")
                    plt.legend(fontsize=18, loc="upper right", bbox_to_anchor=(0.95, 0.9))
                    plt.ylabel("Yaw / $^\circ$", fontsize=40)
                    plt.xlabel("Time / s", fontsize=40)

                    # plt.title("Turn Identification Using Yaw Angle", fontsize=30)
                    plt.xlim(3, 9.5)
                    plt.ylim(-17, 185)
                    plt.xticks(fontsize=20)
                    plt.yticks(fontsize=20)
                    print(plt.yticks())
                    plt.show()

                    # # Plotting
                    # pitch = ahrs.orientation_euler[offset:, 1]
                    # yaw = np.unwrap(ahrs.orientation_euler[offset:, 0], 180)
                    # fig, ax = plt.subplots(2, 1, sharex=True)
                    # x = np.linspace(0, 1 * (len(ahrs.orientation_euler)-offset), len(ahrs.orientation_euler)-offset)
                    # ax[0].plot(x, yaw, label="Yaw")
                    # ax[0].fill_betweenx(yaw, int(turnsIdxs[0]), int(turnsIdxs[1]), facecolor='green', alpha=.5, label="Turn")
                    # ax[0].legend()
                    # ax[0].set_ylabel("Yaw")
                    #
                    # ax[1].plot(x, pitch, label="Pitch")
                    # ax[1].vlines([int(turnsIdxs[2])], min(pitch), max(pitch), color="g", linestyle="--", label="End Prep")
                    # ax[1].vlines([firstStepIdx], min(pitch), max(pitch), color="r", linestyle="--", label="First Step")
                    # ax[1].legend()
                    # ax[1].set_ylabel("Yaw")
                    # ax[1].set_xlabel("Samples")
                    # plt.suptitle("{} - {} Pitch and Yaw".format(opticalFilename.split(".")[0], sideName))
                    # # plt.savefig("../plots/TUG/Angles/{}-{} Angles.png".format(opticalFilename.split(".")[0], sideName))
                    # plt.close()
                    # # plt.show()
                    #
                    # # Show the differential and plot a moving average
                    # N = 80  # len(yawDiffs)
                    # yawDiffs = np.abs(np.diff(lp_filter(yaw, 5)))
                    # yawDiffs = np.convolve(yawDiffs, np.ones(N) / N, mode='valid')
                    # plt.plot(x[:len(yawDiffs)], yawDiffs)
                    # plt.vlines([int(turnsIdxs[0]), int(turnsIdxs[1])], min(yawDiffs), max(yawDiffs))
                    #
                    # # estimate turns
                    # startYawThresh = 15  # minimum yaw angle of IMU allowed for turn to have started
                    # endYawThresh = np.max(np.abs(yaw[int(N/2):int(0.55*len(yaw))])) - 5
                    # print("endYawThresh: ", endYawThresh)
                    # turningPointStartMask = (yawDiffs > 0.3) & (np.abs(yaw[int(N/2):-int(N/2)]) > startYawThresh)
                    # turningPointEndMask = (np.abs(yaw[int(N/2):-int(N/2)]) > endYawThresh)
                    #
                    # # crop out start/end where definitely not turning (avoids confusion
                    # firstQuarter = int(len(turningPointStartMask) * 0.25)
                    # secondQuarter = int(len(turningPointStartMask) * 0.5)
                    # turningPointStartMask = turningPointStartMask[firstQuarter:secondQuarter]
                    #
                    # turningPointStart = first_nonzero(turningPointStartMask, axis=0) + int(N/2) + firstQuarter
                    # turningPointEnd = first_nonzero(turningPointEndMask, axis=0) + int(N/2) #+ secondQuarter
                    # turningEventsDF.loc[opticalFilename, ["StartTurnIdx", "EndTurnIdx"]] = [int(turningPointStart), int(turningPointEnd)]
                    # turningEventsDF.to_csv("../utils/info/TUG Turning Events from IMU.csv")

                    # plt.vlines([int(turningPointStart), int(turningPointEnd)], min(yawDiffs), max(yawDiffs), colors="y")
                    # plt.title("Yaw angle vs. estimated turning points")
                    # plt.savefig("../plots/TUG/EstimatedAngles/{}-{}.png".format(opticalFilename.split(".")[0], sideName))
                    # plt.close()
                    # plt.show()

                    # # plot pitch diffs
                    # # Show the differential and plot a moving average
                    # N = 2  # len(pitchDiffs)
                    # pitchDiffs = np.diff(lp_filter(pitch, 5))
                    # pitchDiffs = np.convolve(pitchDiffs, np.ones(N) / N, mode='valid')
                    # plt.plot(x[:len(pitchDiffs)], pitchDiffs)
                    # plt.vlines([firstStepIdx, turnsIdxs[2]], min(pitchDiffs), max(pitchDiffs), colors="y")
                    # plt.figure()
                    # plt.plot(imuDF.loc[offset:, ["AccZlear", "AccXlear"]])
                    # plt.vlines([firstStepIdx, turnsIdxs[2]], min(imuDF["AccZlear"]), max(imuDF["AccZlear"]), colors="y")
                    # # plt.show()
                    # plt.figure()
                    # plt.plot(imuDF.loc[offset:, "GyroYlear"])
                    # plt.vlines([firstStepIdx, turnsIdxs[2]], min(imuDF["GyroYlear"]), max(imuDF["GyroYlear"]), colors="y")
                    # plt.show()
