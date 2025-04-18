import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def calculate_tsps(diffsDF, filename):
    # Format df correctly
    df = diffsDF.loc[(filename, ["GT ICs", "GT FOs", "IMU ICs", "IMU FOs"]), :]
    df.reset_index(inplace=True)
    df = df.drop("Filename", axis=1)
    df.set_index("EventType", inplace=True)
    df = df.T

    # Shift the FOs until they align
    shiftCount = 1  # start at 1 as we do an extra shift at the end
    while (df["GT FOs"] - df["GT ICs"]).mean() < 0:
        df.iloc[:-1, [1, 3]] = df.iloc[1:, [1, 3]]
        shiftCount += 1
    # Shift once more to get stance time (rather than just double support)
    df.iloc[:-1, [1, 3]] = df.iloc[1:, [1, 3]]
    df = df.loc[~pd.isnull(df["GT ICs"]), :]

    # Calculate Stride times
    df["GTStrideTime"] = df["GT ICs"].shift(periods=-2) - df["GT ICs"]
    df["IMUStrideTime"] = df["IMU ICs"].shift(periods=-2) - df["IMU ICs"]
    # Stance
    df["GTStanceTime"] = df["GT FOs"] - df["GT ICs"]
    df["IMUStanceTime"] = df["IMU FOs"] - df["IMU ICs"]
    # Swing (just difference between the two)
    df["GTSwingTime"] = df["GTStrideTime"] - df["GTStanceTime"]
    df["IMUSwingTime"] = df["IMUStrideTime"] - df["IMUStanceTime"]

    # print(df.GTSwingTime[df["GTSwingTime"] < 0])
    if len(df.GTSwingTime[df["GTSwingTime"] < 0]) > 0:
        print(df)
    # print(df)

    df["Filename"] = filename
    df.index.names = ["EventNum"]
    return df


def save_tsps(df, name):
    # Format and save the summary dfs
    df.set_index(["Filename"], inplace=True, append=True)
    df.index.names = ["EventNum", "Filename"]
    df = df.swaplevel()
    df.to_csv(name)
    

if __name__ == '__main__':
    for activityName in ["WalkNod", "TUG"]:
        diffsDF = pd.read_csv('../TUG/New Data/rawDiffs/{}DiffsRawOverallFFT.csv'.format(activityName),
                              index_col=["Filename", "EventType"])
        # diffsDF = pd.read_csv('../TUG/rawDiffs/{}DiffsRawOverallFFT.csv'.format(activityName), index_col=["Filename", "EventType"])

        summaryColNames = ["Filename", 'GT ICs', 'GT FOs', 'IMU ICs', 'IMU FOs', 'GTStrideTime',
               'IMUStrideTime', 'GTStanceTime', 'IMUStanceTime', 'GTSwingTime',
               'IMUSwingTime']

        if activityName == "TUG":
            turnsInfoDF = pd.read_csv("../utils/info/TUG Turning Events from IMU.csv", index_col="Filename")
            # MAIN DF TO STORE EVERYTHING
            overallTSPsTurnsdf = pd.DataFrame(columns=summaryColNames)
            overallTSPsNonTurnsdf = pd.DataFrame(columns=summaryColNames)
        else:
            overallTSPsdf = pd.DataFrame(columns=summaryColNames)

        diffsDF = diffsDF.filter(regex="Event")

        for filename in diffsDF.index.get_level_values(0)[::4]:
            print(filename)
            tspDF = calculate_tsps(diffsDF, filename)

            if activityName == "TUG":
                # Identify turning / non-turning periods
                turnsIdxs = turnsInfoDF.loc[filename, ["StartTurnIdx", "EndTurnIdx"]].astype(int).tolist()
                gtEventDF = diffsDF.loc[(filename, "GT ICs"), :].dropna()  # .drop(["Event1", "Event2"])
                turnsMask = gtEventDF.between(turnsIdxs[0], turnsIdxs[1])
                turnsCols = gtEventDF[turnsMask].index.tolist()
                nonTurnsCols = gtEventDF[~turnsMask].index.tolist()
                # Add to turns and non-turns dfs respectively
                overallTSPsTurnsdf = pd.concat((overallTSPsTurnsdf, tspDF.loc[turnsCols, :]))
                overallTSPsNonTurnsdf = pd.concat((overallTSPsNonTurnsdf, tspDF.loc[nonTurnsCols, :]))
            else:
                overallTSPsdf = pd.concat((overallTSPsdf, tspDF))

        # Format and save the summary dfs
        if activityName == "TUG":
            save_tsps(overallTSPsTurnsdf, "TUG Turns TSPs Summary.csv")
            save_tsps(overallTSPsNonTurnsdf, "TUG Non Turns TSPs Summary.csv")
        else:
            save_tsps(overallTSPsdf, "{} TSPs Summary.csv".format(activityName))

