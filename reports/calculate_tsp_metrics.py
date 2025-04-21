import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils.participant_info_utils import get_non_typical_participant_nums
import seaborn as sns


def format_tsps_df(activityName, algorithm):
    # Load preformatted df
    if algorithm == "Diao":
        tspsDF = pd.read_csv("../data/Diao/TSPs/{} Diao TSPs Summary.csv".format(activityName))
    elif algorithm == "TP-EAR":
        tspsDF = pd.read_csv("../data/TP-EAR/TSPs/{} TP-EAR TSPs Summary.csv".format(activityName))
    else:
        raise ValueError("Algorithm must be Diao or TP-EAR.")

    # Find the difference between IMU value and GT
    for tspName in ["Stride", "Stance", "Swing"]:
        tspsDF["{}Diffs".format(tspName)] = (tspsDF["IMU{}Time".format(tspName)] - tspsDF["GT{}Time".format(tspName)])

    # Label typical vs non-typical participants
    nonTypicalIDs = get_non_typical_participant_nums()
    subjectNums = np.array([int(x[3:5]) for x in tspsDF["Filename"].values])

    # Split into Typical and Non Typical df
    nonTypicalDF = tspsDF.loc[np.isin(subjectNums, nonTypicalIDs), :]
    typicalDF = tspsDF.loc[~np.isin(subjectNums, nonTypicalIDs), :]

    return typicalDF, nonTypicalDF


def calculate_tsp_metrics_by_state(typicalDF, nonTypicalDF, dataSource):
    """
    Create a table containing mean and standard deviation for the selected data source.
    :param typicalDF:
    :return: typicalArr, nonTypicalArr
    """
    if dataSource == "IMU":
        eventStr = "IMU{}Time"
    elif dataSource == "GT":
        eventStr = "GT{}Time"
    elif dataSource == "Diffs":
        eventStr = "{}Diffs"
    else:
        raise ("Unrecognized data source: {}".format(dataSource))

    typicalArr = np.zeros(3, dtype="object")
    nonTypicalArr = np.zeros_like(typicalArr)

    for i, tspName in enumerate(["Stride", "Stance", "Swing"]):
        # Calculate mean, std and range
        typicalMeanVal = typicalDF.loc[:, eventStr.format(tspName)].abs().mean() * 10
        typicalStdVal = typicalDF.loc[:, eventStr.format(tspName)].abs().std() * 10
        nonTypicalMeanVal = nonTypicalDF.loc[:, eventStr.format(tspName)].abs().mean() * 10
        nonTypicalStdVal = nonTypicalDF.loc[:, eventStr.format(tspName)].abs().std() * 10

        plt.plot(typicalDF.loc[:, eventStr.format(tspName)].abs() * 10)
        plt.axhline(typicalMeanVal, color="r")
        plt.title("abs distribution for Typical {} {}".format(tspName, dataSource))
        plt.show()

        sns.boxplot(x=1, y=typicalDF.loc[:, eventStr.format(tspName)].abs() * 10)
        plt.show()

        # check the csv
        typicalDF.loc[:, eventStr.format(tspName)].abs().mul(10).to_csv("abs distribution for Typical {} {}.csv".format(tspName, dataSource))

        # print("--- {} Time {} Typical ---\nMean: {:2.1f}±{:2.1f}".format(tspName, algorithm, typicalMeanVal, typicalStdVal))
        # print("--- {} Time {} Non Typical ---\nMean: {:2.1f}±{:2.1f}".format(tspName, algorithm, nonTypicalMeanVal,
        #                                                                  nonTypicalStdVal))
        typicalArr[i] = "{:2.1f}±{:2.1f}".format(typicalMeanVal, typicalStdVal)
        nonTypicalArr[i] = "{:2.1f}±{:2.1f}".format(nonTypicalMeanVal, nonTypicalStdVal)
    return typicalArr, nonTypicalArr


if __name__ == "__main__":
    columns = pd.MultiIndex.from_tuples([('Walk', 'Diao'), ('Walk', 'TP-EAR'),
                                         ('WalkNod', 'Diao'), ('WalkNod', 'TP-EAR'),
                                         ('WalkShake', 'Diao'), ('WalkShake', 'TP-EAR'),
                                         ('TUG Turns', 'Diao'), ('TUG Turns', 'TP-EAR'),
                                         ('TUG Non Turns', 'Diao'), ('TUG Non Turns', 'TP-EAR')])

    typicalTable = pd.DataFrame(columns=columns)
    nonTypicalTable = pd.DataFrame(columns=columns)
    for activityName in ["Walk", "WalkShake", "WalkNod", "TUG Turns", "TUG Non Turns"]:
        for algorithm in ["Diao", "TP-EAR"]:
            print(activityName + "\n------------")
            typicalDF, nonTypicalDF = format_tsps_df(activityName, algorithm)
            print(typicalDF)
            typicalTable[(activityName, algorithm)], nonTypicalTable[(activityName, algorithm)] = calculate_tsp_metrics_by_state(typicalDF, nonTypicalDF, "Diffs")

    typicalTable.set_index(pd.Index(data=["Stride (ms)", "Stance (ms)", "Swing (ms)"]), inplace=True)
    nonTypicalTable.set_index(pd.Index(data=["Stride (ms)", "Stance (ms)", "Swing (ms)"]), inplace=True)
    typicalTable.to_csv("TSPs/Difference vs Ground Truth/TSP Difference by Algorithm (Typical).csv".format(activityName))
    nonTypicalTable.to_csv("TSPs/Difference vs Ground Truth/TSP Difference by Algorithm (Non Typical).csv".format(activityName))

