import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils.participant_info_utils import get_non_typical_participant_nums
import seaborn as sns


def format_tsps_df(activityName, isNew=False):
    # Load preformatted df
    if isNew:
        tspsDF = pd.read_csv("NewAlgorithmTSPs/{} TSPs Summary.csv".format(activityName))
    else:
        tspsDF = pd.read_csv("TSPs/{} TSPs Summary.csv".format(activityName))

    # Label typical vs non-typical participants
    nonTypicalIDs = get_non_typical_participant_nums()
    subjectNums = np.array([int(x[3:5]) for x in tspsDF["Filename"].values])
    tspsDF["State"] = "Typical"
    tspsDF.loc[np.isin(subjectNums, nonTypicalIDs), "State"] = "Non-Typical"
    tspsDF["State"] = tspsDF["State"].astype("category")

    for tspName in ["Stride", "Stance", "Swing"]:
        tspsDF["{}Diffs".format(tspName)] = (tspsDF["GT{}Time".format(tspName)] - tspsDF["IMU{}Time".format(tspName)])
        # print(tspsDF["{}Diffs".format(tspName)])

    return tspsDF


def calculate_tsp_metrics_by_state(tspsDF, dataSource):
    """
    Create a boxplot for stride, stance and swing each on their own plot.
    Typical vs non-typical participants are plotted for each event.
    :param tspsDF:
    :param plotName:
    :param save:
    :return:
    """
    if dataSource == "IMU":
        eventStr = "IMU{}Time"
        title = "Time Calculated with IMUs"
    elif dataSource == "GT":
        eventStr = "GT{}Time"
        title = "Time Ground Truth"
    elif dataSource == "Diffs":
        eventStr = "{}Diffs"
        title = "Time Difference between IMU and Ground Truth"
    else:
        raise ("Unrecognized data source: {}".format(dataSource))

    arr = np.zeros((3, 2), dtype="object")
    for i, tspName in enumerate(["Stride", "Stance", "Swing"]):
        for j, state in enumerate(["Typical", "Non-Typical"]):
            # Calculate mean, std and range
            meanVal = tspsDF.loc[tspsDF.State==state, eventStr.format(tspName)].abs().mean() * 10
            stdVal = tspsDF.loc[tspsDF.State==state, eventStr.format(tspName)].abs().std() * 10

            print("--- {} Time {} ---\nMean: {:2.1f}±{:2.1f}".format(tspName, state, meanVal, stdVal))
            arr[i, j] = "{:2.1f}±{:2.1f}".format(meanVal, stdVal)
    print(arr)
    return arr


if __name__ == "__main__":
    columns = pd.MultiIndex.from_tuples([('Walk', 'Typical'), ('Walk', 'Non-Typical'),
                                         ('WalkNod', 'Typical'), ('WalkNod', 'Non-Typical'),
                                         ('WalkShake', 'Typical'), ('WalkShake', 'Non-Typical'),
                                         ('TUG Turns', 'Typical'), ('TUG Turns', 'Non-Typical'),
                                         ('TUG Non Turns', 'Typical'), ('TUG Non Turns', 'Non-Typical')])

    table = pd.DataFrame(columns=columns)
    print(table)
    for activityName in ["Walk", "WalkShake", "WalkNod", "TUG Turns", "TUG Non Turns"]:
    # for activityName in ["TUG Turns", "TUG Non Turns"]:
        # print(table[[(activityName, "Typical"), (activityName, "Non-Typical")]])
        print(activityName + "\n------------")
        # table[[(activityName, "Typical"), (activityName, "Non-Typical")]] = calculate_tsp_metrics_by_state(format_tsps_df(activityName), "Diffs")
        table[[(activityName, "Typical"), (activityName, "Non-Typical")]] = calculate_tsp_metrics_by_state(
            format_tsps_df(activityName, isNew=True), "Diffs")
        print("\n")
    table.set_index(pd.Index(data=["Stride (ms)", "Stance (ms)", "Swing (ms)"]), inplace=True)  #, name="Event"
    print(table)
    table.to_csv("New TSP Differences.csv")

