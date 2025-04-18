import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.signal import resample_poly
from utils.orientation_utils import rearrange_local_to_NED_lsm6dsox
from utils.data_manipulation_utils import set_start_time_to_zero


def fill_imu_time_gaps(subjectNum: int, activity: str, dataDir: str = "../data/",
                       plot: bool = False, save: bool = False, manualGapFill: bool = False):
    """ Fill in missed samples in the IMU data for given subject and activities
    using linear interpolation then optionally them back to CSV.
    Also deletes any corrupted files (i.e. more than 3 consecutive missed frames). """
    activityDir = os.path.join(dataDir, "TF_{}".format(str(subjectNum).zfill(2)), activity)
    # keep track of relative orientations of each sensor

    for file in os.listdir(activityDir):
        print(file)
        # create structures to hold the data
        df = pd.read_csv(os.path.join(activityDir, file))

        # handle unorthodox sensor placements
        if subjectNum in [13, 29]:
            tmp = df[
                ["AccXpocket", "AccYpocket", "AccZpocket", "GyroXpocket", "GyroYpocket", "GyroZpocket", "MagXpocket",
                 "MagYpocket", "MagZpocket", "AccXchest", "AccYchest", "AccZchest", "GyroXchest", "GyroYchest",
                 "GyroZchest", "MagXchest", "MagYchest", "MagZchest"]]
            df[["AccXpocket", "AccYpocket", "AccZpocket", "GyroXpocket", "GyroYpocket", "GyroZpocket", "MagXpocket",
                "MagYpocket", "MagZpocket", "AccXchest", "AccYchest", "AccZchest", "GyroXchest", "GyroYchest",
                "GyroZchest", "MagXchest", "MagYchest", "MagZchest"]] = df[
                ["AccXlear", "AccYlear", "AccZlear", "GyroXlear", "GyroYlear", "GyroZlear", "MagXlear", "MagYlear",
                 "MagZlear", "AccXrear", "AccYrear", "AccZrear", "GyroXrear", "GyroYrear", "GyroZrear", "MagXrear",
                 "MagYrear", "MagZrear"]]
            df[["AccXlear", "AccYlear", "AccZlear", "GyroXlear", "GyroYlear", "GyroZlear", "MagXlear", "MagYlear",
                "MagZlear", "AccXrear", "AccYrear", "AccZrear", "GyroXrear", "GyroYrear", "GyroZrear", "MagXrear",
                "MagYrear", "MagZrear"]] = tmp
        elif subjectNum in [1, 2]:
            # swap pocket and chest
            df[["AccXpocket", "AccYpocket", "AccZpocket", "GyroXpocket", "GyroYpocket", "GyroZpocket", "MagXpocket",
                "MagYpocket",
                "MagZpocket", "AccXchest", "AccYchest", "AccZchest", "GyroXchest", "GyroYchest", "GyroZchest",
                "MagXchest",
                "MagYchest", "MagZchest"]] = df[
                ["AccXchest", "AccYchest", "AccZchest", "GyroXchest", "GyroYchest", "GyroZchest", "MagXchest",
                 "MagYchest", "MagZchest", "AccXpocket", "AccYpocket", "AccZpocket", "GyroXpocket", "GyroYpocket",
                 "GyroZpocket", "MagXpocket", "MagYpocket",
                 "MagZpocket"]]
        elif subjectNum in [68, 69]:
            # Swap left and right ear IMUs
            df[["AccXlear", "AccYlear", "AccZlear", "GyroXlear", "GyroYlear", "GyroZlear", "MagXlear",
                "MagYlear",
                "MagZlear", "AccXrear", "AccYrear", "AccZrear", "GyroXrear", "GyroYrear", "GyroZrear",
                "MagXrear",
                "MagYrear", "MagZrear"]] = df[
                ["AccXrear", "AccYrear", "AccZrear", "GyroXrear", "GyroYrear", "GyroZrear", "MagXrear",
                 "MagYrear", "MagZrear", "AccXlear", "AccYlear", "AccZlear", "GyroXlear", "GyroYlear",
                 "GyroZlear", "MagXlear", "MagYlear",
                 "MagZlear"]]
        # elif subjectNum in [3, 4, 5, 6]:
        #     # swap left and right
        #     df[["AccXlear", "AccYlear", "AccZlear", "GyroXlear", "GyroYlear", "GyroZlear", "MagXlear",
        #         "MagYlear",
        #         "MagZlear", "AccXrear", "AccYrear", "AccZrear", "GyroXrear", "GyroYrear", "GyroZrear",
        #         "MagXrear",
        #         "MagYrear", "MagZrear"]] = df[
        #         ["AccXrear", "AccYrear", "AccZrear", "GyroXrear", "GyroYrear", "GyroZrear", "MagXrear",
        #          "MagYrear", "MagZrear", "AccXlear", "AccYlear", "AccZlear", "GyroXlear", "GyroYlear",
        #          "GyroZlear", "MagXlear", "MagYlear",
        #          "MagZlear"]]
            

        dfArr = df.to_numpy()

        # don't save the file if there's been a major timing corruption
        dataCorruptedFlag = False
        timeDiffs = df.AccATime.diff().dropna().mod(2**16)

        # check there's been no time drift
        if len(timeDiffs[np.logical_and(timeDiffs % 10 != 0, timeDiffs > 0)]) > 0:
            print(file)
            print(timeDiffs[timeDiffs % 10 != 0])
            print(timeDiffs[timeDiffs % 10 != 0].div(10).round().mul(10))

        # assuming time drift is minor (e.g. 11.0 followed by 9.0) just correct by rounding both
        # timeDiffs[timeDiffs % 10 != 0] = timeDiffs[timeDiffs % 10 != 0].div(10).round().mul(10)
        # check that gaps are sufficiently small and rare
        numMissedSamples = int(timeDiffs[timeDiffs > 10].sub(10).sum() / 10)
        print("numMissedSamples: ", numMissedSamples)
        if numMissedSamples / len(df) > 0.2 and int(file.split(".")[0][-1]) != 1:
            dataCorruptedFlag = True
            print(file, numMissedSamples, timeDiffs.max() / 10)
            print(file + " is too corrupted - skipping this one.")

        # # create new array with correct indices - we can fill this using our a priori knowledge
        if not dataCorruptedFlag:
            # create new Array with indices available to insert missing values
            newArr = np.empty((df.shape[0] + numMissedSamples, df.shape[1]))
            newArr[:, :] = np.NaN  # allows us to use pd.interpolate()

            # reindex our existing dataframe using the cumsum (takes account varying gap lengths)
            alreadyInputtedIndices = timeDiffs.cumsum().div(10).astype(int).to_list()
            alreadyInputtedIndices.insert(0, 0)
            newArr[alreadyInputtedIndices, :] = dfArr

            # convert this back to a pd df
            newDF = pd.DataFrame(newArr, columns=df.columns)
            newDF["IsInterpolated"] = newDF.AccATime.isna()  # column to hold gaps where we will interpolate
            # interpolate gaps in new df if gap < 3 otherwise leave as NaN
            # if numMissedSamples > 0:
            # find the indices at which there are gaps
            gapIdxList = newDF[newDF.IsInterpolated].index.to_list()
            if manualGapFill:
                # create an array to store the gap sizes affiliated with each group
                gapSizeCount = 1
                gapIdxCountList = []
                for i in range(1, len(gapIdxList)):
                    if gapIdxList[i] - 1 == gapIdxList[i-1]:
                        gapSizeCount += 1
                    else:
                        gapIdxCountList.append([gapIdxList[i-gapSizeCount], gapSizeCount])
                        gapSizeCount = 0
                gapIdxCountList.append([gapIdxList[-(gapSizeCount+1)], gapSizeCount+1])
                gapIdxCountArr = np.array(gapIdxCountList) # assign to a np array for easier access
                # interpolate gaps 2 or less in a row
                # idxToFill = gapIdxCountArr[gapIdxCountArr[:, 1] < 3]
                # missing = idxToFill
                missing = newDF["AccATime"].isna()
                newDF["IsInterpolated"] = missing
                dfTraining = newDF[~missing]
                dfMissing = newDF[missing]
                # print(df.filter(regex="Time|Frame"))
                for col in df.columns:
                    f = interp1d(dfTraining.index.to_list(), dfTraining[col])
                    newDF.loc[dfMissing.index.to_list(), col] = f(dfMissing.index.to_list())
            else:
                newDF = newDF.interpolate(method='linear')

                # plt.plot(df[["AccXpocket", "AccYpocket", "AccZpocket"]])
                # plt.legend(["X", "Y", "Z"])
                # plt.show()

                # sensorPlacementArr[subjectNum] = np.sign(newDF.loc[0, ["AccXlear", "AccXrear", "AccYchest", "AccZpocket"]])

                # Convert the data to NED format
                if subjectNum not in [12, 13, 63]:
                    newDF = rearrange_local_to_NED_lsm6dsox(newDF, subjectNum, check=True)
                else:
                    newDF = rearrange_local_to_NED_lsm6dsox(newDF, subjectNum, check=False)

            # Condense the timestamp data and reindex to start at time t=0
            reducedCols = [x for x in newDF.columns if ("Time" not in x and "Frame" not in x)]
            reducedCols.insert(0, "AccATime")
            newDF = newDF[reducedCols].rename(columns={"AccATime": "Time"})
            newDF = set_start_time_to_zero(newDF)


            # visualise this effect
            if plot:
                # # plot the vals either side of gap as well otherwise the joining line doesn't show
                # gapOnlyDF = newDF.where(newDF.IsInterpolated | newDF.shift(1).IsInterpolated | newDF.shift(-1).IsInterpolated,
                #                         np.NaN)
                # noGapDF = newDF.where(~newDF.IsInterpolated, np.NaN)
                # plt.plot(gapOnlyDF.AccZlear, color='r', label="Interpolations")
                # plt.plot(noGapDF.AccZlear, color='b', label="Raw data")
                # plt.title(file)
                # plt.show()
                
                # Compare right and left
                from itertools import compress
                dfL = newDF[list(compress(newDF.columns, newDF.columns.str.contains("lear")))]
                dfR = newDF[list(compress(newDF.columns, newDF.columns.str.contains("rear")))]
                plt.plot(dfL[["GyroXlear", "GyroYlear", "GyroZlear"]])  # "AccXlear", "AccYlear",
                plt.plot(dfR[["GyroXrear", "GyroYrear", "GyroZrear"]])  # "AccXrear", "AccYrear",
                plt.title("TF_{} Gyro".format(subjectNum))
                plt.show()
                plt.plot(dfL[["AccXlear", "AccYlear", "AccZlear"]])  # "AccXlear", "AccYlear",
                plt.plot(dfR[["AccXrear", "AccYrear", "AccZrear"]])  # "AccXrear", "AccYrear",
                plt.title("TF_{} Acc".format(subjectNum))
                plt.show()
                plt.plot(dfL[["MagXlear", "MagYlear", "MagZlear"]])  # "AccXlear", "AccYlear",
                plt.plot(dfR[["MagXrear", "MagYrear", "MagZrear"]])  # "AccXrear", "AccYrear",
                plt.title("TF_{} Mag".format(subjectNum))
                plt.show()
            if save:
                if activity == "Static":
                    print("Hi")
                    newDF.to_csv(os.path.join("data/statics/TF_{}.csv".format(str(subjectNum).zfill(2), file)),
                                 index=False)
                else:
                    newDF.to_csv(os.path.join("data/TF_{}/{}.csv".format(str(subjectNum).zfill(2), file.split(".")[0])), index=False)


# Function test
if __name__ == "__main__":
    # directories for accessing raw data
    dataDir = os.path.normpath("C:/Users/teri-/Documents/Gait Experiment Data (Actual Trial)")
    # activitiesList = ["Sit2Stand", "Reach", "Stand2Sit", "TUG"]
    # activitiesList = ["Walk", "WalkNod", "WalkShake"]
    activitiesList = ["WalkSlow"]
    # activitiesList = ["Static"]
    # populate project data folder if not already present
    if not os.path.exists("data/"):
        os.mkdir("data/")
    if len(os.listdir("data")) == 0:
        os.mkdir("data/statics")
        for subjectNum in range(1, 70):
            os.mkdir("data/TF_{}".format(str(subjectNum).zfill(2)))

    sensorPlacementArr = np.zeros((70, 4))

    # loop through participants
    for subjectNum in [x for x in range(1, 72) if x not in [100]]:
        for activity in activitiesList:
            fill_imu_time_gaps(subjectNum,
                               activity,
                               dataDir="C:/Users/teri-/Documents/Gait Experiment Data (Actual Trial)/",
                               plot=False,
                               save=True,
                               manualGapFill=False)

    # sensorPlacementDF = pd.DataFrame(sensorPlacementArr, columns=["Left Ear", "Right Ear", "Chest", "Pocket"])
    # sensorPlacementDF["Participant Number"] = [int(x) for x in range(0, 68)]
    # sensorPlacementDF = sensorPlacementDF[["Participant Number", "Left Ear", "Right Ear", "Chest", "Pocket"]]
    # sensorPlacementDF.to_csv("SensorPlacementSigns.csv", index=False)

    # dataDir = os.path.normpath("C:/Users/teri-/Downloads/shit2stand/shit2stand/")
    # activity = "Sit2Stand"
    # fill_imu_time_gaps(100,
    #                    activity,
    #                    dataDir=dataDir,
    #                    plot=True,
    #                    save=True,
    #                    manualGapFill=False)

