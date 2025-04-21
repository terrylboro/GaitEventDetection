import pandas as pd
import numpy as np
from utils.participant_info_utils import get_non_typical_participant_nums


def describe_tsps(plotName1, plotName2, dataSource, tsParam):
    """
    Compare parameter for two activities on a single plot.
    Typical vs non-typical participants are plotted for each event.
    :param tspsDF:
    :param dataSource:
    :param title:
    :param save:
    :return:
    """
    if dataSource == "IMU":
        eventStr = "IMU{}Time"
    elif dataSource == "GT":
        eventStr = "GT{}Time"
    elif dataSource == "Diffs":
        eventStr = "{}Diffs"
    else:
        raise ("Unrecognized data source: {}".format(dataSource))

    nonTypicalIDs = get_non_typical_participant_nums()  # used to split typical/non-typical

    for state in ["Typical", "Non-Typical"]:
        # Reformat the df to fit boxplot format
        boxDF = pd.DataFrame(columns=["Activity", "EventType", "Value"])
        for activityTitle, activityName in zip(["Walk", "WalkNod", "WalkShake", "TUG Turns"], ["Walk", "WalkV", "WalkH", "Turn"]):
            tspsDF1 = pd.read_csv("../data/Diao/TSPs/{} Diao TSPs Summary.csv".format(activityTitle))
            tspsDF2 = pd.read_csv("../data/TP-EAR/TSPs/{} TP-EAR TSPs Summary.csv".format(activityTitle))

            # Split into typical and non-typical
            subjectNums = np.array([int(x[3:5]) for x in tspsDF1["Filename"].values])
            stateMask = np.isin(subjectNums, nonTypicalIDs) if state == "Non-Typical" else ~np.isin(subjectNums, nonTypicalIDs)
            tspsDF1 = tspsDF1.loc[stateMask, :]
            tspsDF2 = tspsDF2.loc[stateMask, :]

            # Fill df with chosen TSP data for each algorithm
            for tspsDF, algorithmName in zip([tspsDF1, tspsDF2], [plotName1, plotName2]):
                # Calculate difference between IMU and GT
                tspsDF["{}Diffs".format(tsParam)] = (
                            tspsDF["IMU{}Time".format(tsParam)] - tspsDF["GT{}Time".format(tsParam)])
                # Compile all the diffs into a df to be plotted
                for cycle in tspsDF.index.values:
                    eventArr = np.full((1, 4), fill_value=np.nan, dtype="object")
                    for i, eventName in enumerate([tsParam]):
                        eventArr[i] = [activityName, algorithmName, eventName, tspsDF.at[cycle, eventStr.format(eventName)] * 10]
                    # Add to overall df
                    boxDF = pd.concat((boxDF, pd.DataFrame(data=eventArr, columns=["Activity", "Algorithm", "EventType", "Value"])))

        # Ensure the types are categorical and indexed correctly then plot
        boxDF = boxDF.astype(dtype={"Activity": "category", "Algorithm": "category", "EventType": "category", "Value": "float"})
        boxDF.Activity = pd.Categorical(boxDF.Activity, categories=["Walk", "WalkV", "WalkH", "Turn"], ordered=True)
        boxDF.Algorithm = pd.Categorical(boxDF.Algorithm, categories=["Diao", "TP-EAR"])

        boxDF.Value = boxDF.Value.astype(float)
        boxDF["AbsValue"] = boxDF.Value.abs().astype(float)
        boxDF = boxDF.reset_index()

        for isSigned in [True, False]:
            describeArr = np.zeros((1, 10))
            for activity in ["Walk", "WalkV", "WalkH", "Turn"]:
                for algorithm in ["Diao", "TP-EAR"]:
                    arr = np.zeros(10, dtype="object")
                    arr[0:2] = [activity, algorithm]
                    if isSigned:
                        arr[2:] = boxDF[(boxDF.Activity==activity) & (boxDF.Algorithm==algorithm)].describe().Value.to_numpy()
                    else:
                        arr[2:] = boxDF[(boxDF.Activity == activity) & (boxDF.Algorithm == algorithm)].describe().AbsValue.to_numpy()
                    describeArr = np.vstack((describeArr, arr))

            describeDF = pd.DataFrame(describeArr[1:],
                                      columns=["Activity", "Algorithm", "Count", "Mean", "Std", "Min", "25%", "50%", "75%",
                                               "Max"])

            if isSigned:
                describeDF.to_csv(
                    "../reports/TSPs/Difference vs Ground Truth Description/{}{}Signed.csv".format(state, tsParam),
                    index=False)
            else:
                describeDF.to_csv("../reports/TSPs/Difference vs Ground Truth Description/{}{}Unsigned.csv".format(state, tsParam), index=False)


if __name__ == '__main__':
    # Compare turning vs non-turning
    for source in ["Diffs"]:
        for isAbsolute in [True, False]:
            for param in ["Stride", "Stance", "Swing"]:
                describe_tsps("Diao", "TP-EAR", source, tsParam=param)





