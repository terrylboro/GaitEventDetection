def rearrange_local_to_NED_lsm6dsox(df, subjectNum, check=True):
    """
    Reorder the IMU data into NED column format. Contains some checks for sensor misalignment.
    Assumes the LIS3MDL (magnetometer) is aligned the same way as the LSM6DSOX in the IC.
    Right Ear: +X is down, +Y is north, +Z is east
    Left Ear: +X is up, +Y is north, +Z is west
    Chest: +X is east, +Y is up, +Z is north
    Pocket: +X is down, +Y is west, +Z is north
    NED: +X is north, +Y is east, +Z is down
    :param df: Pandas DataFrame containing raw or lightly processed IMU data in sensor frame
    :return: Reordered Pandas DataFrame in NED column format.
    """
    import pandas as pd
    # left ear
    # firstly check which way the IMU is flipped by using gravity
    if df["AccXlear"].mean() < 0:
        if subjectNum in [5, 68, 69, 70, 71]:  # Include 5 for trialNum < 7
            df['AccXlear'], df['AccYlear'], df['AccZlear'] = - df['AccYlear'], - df['AccZlear'], - df['AccXlear']
            df['GyroXlear'], df['GyroYlear'], df['GyroZlear'] = df['GyroYlear'], df['GyroZlear'], df['GyroXlear']
            df['MagXlear'], df['MagYlear'], df['MagZlear'] = - df['MagYlear'], - df['MagZlear'], - df['MagXlear']
        else:
            df['AccXlear'], df['AccYlear'], df['AccZlear'] = df['AccYlear'], df['AccZlear'], - df['AccXlear']
            df['GyroXlear'], df['GyroYlear'], df['GyroZlear'] = - df['GyroYlear'], - df['GyroZlear'], df['GyroXlear']
            df['MagXlear'], df['MagYlear'], df['MagZlear'] = df['MagYlear'], df['MagZlear'], - df['MagXlear']
    else:
        if subjectNum in [1, 2]: # 69
            df['AccXlear'], df['AccYlear'], df['AccZlear'] = - df['AccYlear'], df['AccZlear'], df['AccXlear']
            df['GyroXlear'], df['GyroYlear'], df['GyroZlear'] = df['GyroYlear'], - df['GyroZlear'], -df['GyroXlear']
            df['MagXlear'], df['MagYlear'], df['MagZlear'] = - df['MagYlear'], df['MagZlear'], df['MagXlear']
        else:
            df['AccXlear'], df['AccYlear'], df['AccZlear'] = df['AccYlear'], - df['AccZlear'], df['AccXlear']
            df['GyroXlear'], df['GyroYlear'], df['GyroZlear'] = - df['GyroYlear'], df['GyroZlear'], -df['GyroXlear']
            df['MagXlear'], df['MagYlear'], df['MagZlear'] = df['MagYlear'], - df['MagZlear'], df['MagXlear']

    # correct using sit-to-stand
    # df['AccXlear'], df['AccYlear'], df['AccZlear'] = df['AccYlear'], - df['AccZlear'], - df['AccXlear']
    # df['GyroXlear'], df['GyroYlear'], df['GyroZlear'] = -df['GyroYlear'], df['GyroZlear'], df['GyroXlear']
    # df['MagXlear'], df['MagYlear'], df['MagZlear'] = df['MagYlear'], - df['MagZlear'], - df['MagXlear']

    # right ear
    if df["AccXrear"].mean() < 0:
        if subjectNum in [1, 2, 3, 4]:  #69 5
            print("Found mix up in sides!")
            df['AccXrear'], df['AccYrear'], df['AccZrear'] = - df['AccYrear'], -df['AccZrear'], - df['AccXrear']
            df['GyroXrear'], df['GyroYrear'], df['GyroZrear'] = df['GyroYrear'], df['GyroZrear'], df['GyroXrear']
            df['MagXrear'], df['MagYrear'], df['MagZrear'] = - df['MagYrear'], -df['MagZrear'], - df['MagXrear']
        else:
            # works for TF_01 up to TF_05
            df['AccXrear'], df['AccYrear'], df['AccZrear'] = df['AccYrear'], df['AccZrear'], - df['AccXrear']
            df['GyroXrear'], df['GyroYrear'], df['GyroZrear'] = - df['GyroYrear'], - df['GyroZrear'], df['GyroXrear']
            df['MagXrear'], df['MagYrear'], df['MagZrear'] = df['MagYrear'], df['MagZrear'], - df['MagXrear']

    else:
        # if subjectNum == 6:
        #     print("found subjectNum 6")
        #     df['AccXrear'], df['AccYrear'], df['AccZrear'] = - df['AccYrear'], - df['AccZrear'], df['AccXrear']
        #     df['GyroXrear'], df['GyroYrear'], df['GyroZrear'] = df['GyroYrear'], - df['GyroZrear'], - df['GyroXrear']
        #     df['MagXrear'], df['MagYrear'], df['MagZrear'] = - df['MagYrear'], - df['MagZrear'], df['MagXrear']
        # else:
        if 71 > subjectNum > 69:
            print("Processing with minus!")
            df['AccXrear'], df['AccYrear'], df['AccZrear'] = - df['AccYrear'], - df['AccZrear'], df['AccXrear']
            df['GyroXrear'], df['GyroYrear'], df['GyroZrear'] = df['GyroYrear'], df['GyroZrear'], - df['GyroXrear']
            df['MagXrear'], df['MagYrear'], df['MagZrear'] = - df['MagYrear'], - df['MagZrear'], df['MagXrear']
        else:
        # try for TF_06 onwards
            df['AccXrear'], df['AccYrear'], df['AccZrear'] = - df['AccYrear'], df['AccZrear'], df['AccXrear']
            df['GyroXrear'], df['GyroYrear'], df['GyroZrear'] = df['GyroYrear'], - df['GyroZrear'], - df['GyroXrear']
            df['MagXrear'], df['MagYrear'], df['MagZrear'] = - df['MagYrear'], df['MagZrear'], df['MagXrear']

    # df['AccXrear'], df['AccYrear'], df['AccZrear'] = df['AccYrear'], df['AccZrear'], df['AccXrear']
    # df['GyroXrear'], df['GyroYrear'], df['GyroZrear'] = -df['GyroYrear'], -df['GyroZrear'], -df['GyroXrear']
    # df['MagXrear'], df['MagYrear'], df['MagZrear'] = df['MagYrear'], df['MagZrear'], df['MagXrear']

    # # chest
    if df['AccYchest'].mean() < 0:
        df['AccXchest'], df['AccYchest'], df['AccZchest'] = -df['AccZchest'], df['AccXchest'], - df['AccYchest']
        df['GyroXchest'], df['GyroYchest'], df['GyroZchest'] = - df['GyroZchest'], df['GyroXchest'], - df['GyroYchest']
        df['MagXchest'], df['MagYchest'], df['MagZchest'] = -df['MagZchest'], df['MagXchest'], - df['MagYchest']
    else:
        df['AccXchest'], df['AccYchest'], df['AccZchest'] = -df['AccZchest'], -df['AccXchest'], df['AccYchest']
        df['GyroXchest'], df['GyroYchest'], df['GyroZchest'] = df['GyroZchest'], -df['GyroXchest'], - df['GyroYchest']
        df['MagXchest'], df['MagYchest'], df['MagZchest'] = -df['MagZchest'], -df['MagXchest'], df['MagYchest']

    # pocket
    if df['AccXpocket'].mean() > 0:
        df['AccXpocket'], df['AccYpocket'], df['AccZpocket'] = -df['AccZpocket'], df['AccYpocket'], df['AccXpocket']
        df['GyroXpocket'], df['GyroYpocket'], df['GyroZpocket'] = df['GyroZpocket'], df['GyroYpocket'], -df['GyroXpocket']
        df['MagXpocket'], df['MagYpocket'], df['MagZpocket'] = -df['MagZpocket'], df['MagYpocket'], df['MagXpocket']
    else:
        df['AccXpocket'], df['AccYpocket'], df['AccZpocket'] = -df['AccZpocket'], -df['AccYpocket'], -df['AccXpocket']
        df['GyroXpocket'], df['GyroYpocket'], df['GyroZpocket'] = df['GyroZpocket'], df['GyroYpocket'], df['GyroXpocket']
        df['MagXpocket'], df['MagYpocket'], df['MagZpocket'] = -df['MagZpocket'], -df['MagYpocket'], -df['MagXpocket']

    # check that this is all reasonable
    # if check:
        assert (df['AccZlear'].mean() >= 0 and df['AccZlear'].mean() >= df['AccXlear'].mean()), "Left ear rotation is off"
        assert (df['AccZrear'].mean() >= 0 and df['AccZrear'].mean() >= df['AccXrear'].mean()), "Right ear rotation is off"
        assert (df['AccZchest'].mean() >= 0 and df['AccZchest'].mean() >= df['AccXchest'].mean()), "Chest rotation is off"
        # pocket is sometimes zero so >= works best
        # assert (df['AccZpocket'].mean() >= 0 and df['AccZpocket'].mean() >= df['AccXpocket'].mean()), "Pocket rotation is off"

    return df


def calculate_tilt_corrected_linear_acceleration_adapted(df, gVal=9.81):
    from AdaptedMatlabAHRS.AHRS import AHRS
    import numpy as np
    from itertools import compress
    from utils.data_manipulation_utils import calculate_acc_zero
    df = df.dropna()
    if not (len(df.loc[:, "AccXlear"]) == 0 or np.count_nonzero(df.loc[:, "AccXlear"].to_numpy()) == 0):
        dfL = df[list(compress(df.columns, df.columns.str.contains("lear")))]
        dfR = df[list(compress(df.columns, df.columns.str.contains("rear")))]
        dfC = df[list(compress(df.columns, df.columns.str.contains("chest")))]

        # N: 19,177.3 nT	E: 244.9 nT	D: 45,347.7 nT	Total: 49,236.6 nT
        # earthMagVec = np.array([19.177, 0, 10])
        # # earthMagVec = np.array([19.177, 10, 10])
        # # earthMagVec = df.loc[0, ['MagXrear', 'MagYrear', 'MagZrear']]
        # dfL.loc[:, ['MagXlear', 'MagYlear', 'MagZlear']] += (earthMagVec - dfL.loc[:, ['MagXlear', 'MagYlear', 'MagZlear']])
        # dfR.loc[:, ['MagXrear', 'MagYrear', 'MagZrear']] += (earthMagVec - dfR.loc[:, ['MagXrear', 'MagYrear', 'MagZrear']])
        imuMeasurementsL = dfL[["AccXlear", "AccYlear", "AccZlear", "GyroXlear", "GyroYlear", "GyroZlear",
                                "MagXlear", "MagYlear", "MagZlear"]].astype(float).to_numpy()
        imuMeasurementsR = dfR[["AccXrear", "AccYrear", "AccZrear", "GyroXrear", "GyroYrear", "GyroZrear",
                                "MagXrear", "MagYrear", "MagZrear"]].astype(float).to_numpy()
        imuMeasurementsC = dfC[["AccXchest", "AccYchest", "AccZchest", "GyroXchest", "GyroYchest", "GyroZchest",
                                "MagXchest", "MagYchest", "MagZchest"]].astype(float).to_numpy()

        # Calculate the ear orientations
        ahrsL = AHRS(is6axis=True)
        ahrsR = AHRS(is6axis=True)
        ahrsC = AHRS(is6axis=True)
        ahrsL.run(imuMeasurements=imuMeasurementsL, plot=False)
        ahrsR.run(imuMeasurements=imuMeasurementsR, plot=False)
        if not imuMeasurementsC[:, 0].all() == 0:
            ahrsC.run(imuMeasurements=imuMeasurementsC, plot=False)
            ahrsZippedList = zip([ahrsL, ahrsR, ahrsC], ["lear", "rear", "chest"])
        else:
            ahrsZippedList = zip([ahrsL, ahrsR], ["lear", "rear"])


        # Use output orientation to calculate linear acceleration
        linAcc = np.zeros((len(df), 9))
        linAccResolved = np.zeros((len(df), 9))
        gVecArr = np.zeros((len(df), 9))

        for sensorNum, (ahrs, side) in enumerate(ahrsZippedList):
            for i in range(len(df)):
                # apply initial offset correction
                # ahrs gives point rotation so we transpose to get frame rotation
                rotMat = ahrs.rotmat[i]
                gVector = np.matmul(rotMat.T, np.array([0, 0, gVal]).reshape((3, 1)))
                # gVecArr[i, :] = gVector.T

                temp = df.loc[i, ["AccX" + side, "AccY" + side, "AccZ" + side]].to_numpy() - gVector.T

                linAcc[i, (sensorNum*3):((sensorNum+1)*3)] = temp.T.squeeze()  #np.matmul(rotMat.T, temp.T).squeeze()
                linAccResolved[i, (sensorNum*3):((sensorNum+1)*3)] = np.matmul(rotMat, temp.T).squeeze()

        # add in the timestamp and return pitch angle
        return linAccResolved
    else:
        raise ValueError("All values are zero or NaN")
