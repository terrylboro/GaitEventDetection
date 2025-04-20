import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils.participant_info_utils import get_non_typical_participant_nums
import seaborn as sns

def boxplot_compare_activities(tspsDF1, tspsDF2, plotName1, plotName2, dataSource, activityName, save=False):
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
        title = "{} vs {} TSPs Calculated with IMUs".format(plotName1, plotName2)
    elif dataSource == "GT":
        eventStr = "GT{}Time"
        title = "{} vs {} Ground Truth TSPs".format(plotName1, plotName2)
    elif dataSource == "Diffs":
        eventStr = "{}Diffs"
        # title = "{} vs {} TSP Difference between IMU and Ground Truth".format(plotName1, plotName2)
        title = "{} vs {} Difference in TSP between IMU and Ground Truth for {}".format(plotName1, plotName2, activityName)
    else:
        raise ("Unrecognized data source: {}".format(dataSource))

    for state in ["Typical", "Non-Typical"]:
        # Reformat the df to fit boxplot format
        boxDF = pd.DataFrame(columns=["Activity", "EventType", "Value"])
        for tspsDF, activityName in zip([tspsDF1.loc[tspsDF1.State==state, :], tspsDF2.loc[tspsDF2.State==state, :]], [plotName1, plotName2]):
            for cycle in tspsDF.index.values:
                eventArr = np.full((3, 3), fill_value=np.nan, dtype="object")
                for i, eventName in enumerate(["Stride", "Stance", "Swing"]):
                    eventArr[i] = [activityName, eventName, tspsDF.at[cycle, eventStr.format(eventName)] / 100]
                # Add to overall df
                boxDF = pd.concat((boxDF, pd.DataFrame(data=eventArr, columns=["Activity", "EventType", "Value"])))

        # Ensure the types are categorical and indexed correctly then plot
        boxDF = boxDF.astype(dtype={"Activity": "category", "EventType": "category", "Value": "float"})
        boxDF.Activity = pd.Categorical(boxDF.Activity, categories=[plotName1, plotName2], ordered=True)
        boxDF.EventType = pd.Categorical(boxDF.EventType, categories=["Stride", "Stance", "Swing"], ordered=True)
        boxDF.Value = boxDF.Value.astype(float)
        boxDF = boxDF.reset_index()
        sns.boxplot(x="EventType", y="Value", hue="Activity", data=boxDF)
        plt.title(title + "\n{} Participants".format(state))
        plt.xlabel("Parameter")
        plt.ylabel("Time / s")

        if save:
            # plt.savefig("BoxWhisker/{}.png".format("-".join(str(title + "{} Participants".format(state)).split(" "))))
            plt.savefig("NewAlgorithm/CompareAlgorithms/{}.png".format("-".join(str(title + "{} Participants".format(state)).split(" "))))
            plt.close()
        else:
            plt.show()


def boxplot_compare_algorithm_tsps(plotName1, plotName2, dataSource, tsParam, save=False):
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
        title = "{} vs {} TSPs Calculated with IMUs".format(plotName1, plotName2)
    elif dataSource == "GT":
        eventStr = "GT{}Time"
        title = "{} vs {} Ground Truth TSPs".format(plotName1, plotName2)
    elif dataSource == "Diffs":
        eventStr = "{}Diffs"
        # title = "{} vs {} TSP Difference between IMU and Ground Truth".format(plotName1, plotName2)
        title = "{} vs {} {} Time Difference between IMU and Ground Truth".format(plotName1, plotName2, tsParam)
    else:
        raise ("Unrecognized data source: {}".format(dataSource))

    nonTypicalIDs = get_non_typical_participant_nums()  # used to split typical/non-typical

    for state in ["Typical", "Non-Typical"]:
        # Reformat the df to fit boxplot format
        boxDF = pd.DataFrame(columns=["Activity", "EventType", "Value"])
        for activityTitle, activityName in zip(["Walk", "WalkNod", "WalkShake", "TUG Turns"], ["Walk", "WalkV", "WalkH", "Turn"]):
            tspsDF1 = pd.read_csv("../data/Diao/TSPs/{} Diao TSPs Summary.csv".format(activityTitle))
            tspsDF2 = pd.read_csv("../data/TP-EAR/TSPs/{} TP-EAR TSPs Summary.csv".format(activityTitle))

            print("{} - {}".format(activityTitle, state))
            print(tspsDF1[~(tspsDF1['Filename'].isin(tspsDF2['Filename']))])#.drop_duplicates(keep=False))

            # Split into typical and non-typical
            subjectNums = np.array([int(x[3:5]) for x in tspsDF1["Filename"].values])
            stateMask = np.isin(subjectNums, nonTypicalIDs) if state == "Non-Typical" else ~np.isin(subjectNums, nonTypicalIDs)
            # print(stateMask)
            tspsDF1 = tspsDF1.loc[stateMask, :]
            # print(tspsDF1.shape)
            tspsDF2 = tspsDF2.loc[stateMask, :]
            # print(tspsDF2.shape)

            # Fill df with chosen TSP data for each algorithm
            for tspsDF, algorithmName in zip([tspsDF1, tspsDF2], [plotName1, plotName2]):
                # Calculate difference between IMU and GT
                tspsDF["{}Diffs".format(tsParam)] = (
                            tspsDF["IMU{}Time".format(tsParam)] - tspsDF["GT{}Time".format(tsParam)])
                # Compile all the diffs into a df to be plotted
                for cycle in tspsDF.index.values:
                    eventArr = np.full((1, 4), fill_value=np.nan, dtype="object")
                    for i, eventName in enumerate([tsParam]):
                        eventArr[i] = [activityName, algorithmName, eventName, tspsDF.at[cycle, eventStr.format(eventName)] / 100]
                    # Add to overall df
                    boxDF = pd.concat((boxDF, pd.DataFrame(data=eventArr, columns=["Activity", "Algorithm", "EventType", "Value"])))

        # Ensure the types are categorical and indexed correctly then plot
        boxDF = boxDF.astype(dtype={"Activity": "category", "Algorithm": "category", "EventType": "category", "Value": "float"})
        boxDF.Activity = pd.Categorical(boxDF.Activity, categories=["Walk", "WalkV", "WalkH", "Turn"], ordered=True)
        boxDF.Algorithm = pd.Categorical(boxDF.Algorithm, categories=["Diao", "TP-EAR"])
        # boxDF.EventType = pd.Categorical(boxDF.EventType, categories=["Stance"], ordered=True)
        boxDF.Value = boxDF.Value.astype(float)
        boxDF = boxDF.reset_index()

        # Plot the figure
        plt.figure(figsize=(12, 6))
        sns.boxplot(x="Activity", y="Value", hue="Algorithm", data=boxDF)
        # plt.title(title + "\n{} Participants".format(state))
        plt.xticks(size=18)
        plt.yticks(size=12)
        plt.xlabel("Parameter", size=22)
        plt.ylabel("Time / s", size=22)
        plt.tight_layout()
        # plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

        if save:
            # plt.savefig("BoxWhisker/{}.png".format("-".join(str(title + "{} Participants".format(state)).split(" "))))
            plt.savefig("Comparison Plots/{}Comparison{}.png".format(tsParam, state), dpi=300)
            plt.close()
        else:
            plt.show()


if __name__ == '__main__':
    # Plot the walking activities first
    # for activityName, plotName in zip(["Walk", "WalkShake", "WalkNod"], ["Walk", "WalkH", "WalkV"]):
    #     tspsDF = format_tsps_df(activityName, isNew=True)
    #
    #     boxplot_all_tsps_by_state(tspsDF, plotName, "IMU", save=True, isNew=True)
    #     boxplot_all_tsps_by_state(tspsDF, plotName, "GT", save=True, isNew=True)
    #     boxplot_all_tsps_by_state(tspsDF, plotName, "Diffs", save=True, isNew=True)
    #
    #     boxplot_separate_tsps_by_state(tspsDF, plotName, "IMU", save=True, isNew=True)
    #     boxplot_separate_tsps_by_state(tspsDF, plotName, "GT", save=True, isNew=True)
    #     boxplot_separate_tsps_by_state(tspsDF, plotName, "Diffs", save=True, isNew=True)
    #
    #
    # # Process TUG trials
    # for activityName, plotName in zip(["TUG Turns", "TUG Non Turns"], ["TUG Turns", "TUG Non Turns"]):
    #     # tspsDF = format_tsps_df(activityName)
    #     tspsDF = format_tsps_df(activityName, isNew=True)
    #
    #     boxplot_all_tsps_by_state(tspsDF, plotName, "IMU", save=True, isNew=True)
    #     boxplot_all_tsps_by_state(tspsDF, plotName, "GT", save=True, isNew=True)
    #     boxplot_all_tsps_by_state(tspsDF, plotName, "Diffs", save=True, isNew=True)
    #
    #     boxplot_separate_tsps_by_state(tspsDF, plotName, "IMU", save=True, isNew=True)
    #     boxplot_separate_tsps_by_state(tspsDF, plotName, "GT", save=True, isNew=True)
    #     boxplot_separate_tsps_by_state(tspsDF, plotName, "Diffs", save=True, isNew=True)

    # Compare turning vs non-turning
    for source in ["Diffs"]:
        # boxplot_compare_activities(format_tsps_df("TUG Turns"), format_tsps_df("TUG Non Turns"),
        #                            "TUG Turns", "TUG Non Turns",
        #                            source, save=True)

        # boxplot_compare_activities(format_tsps_df("WalkShake", isNew=True), format_tsps_df("WalkShake", isNew=False),
        #                            "New", "Diao",
        #                            source, activityName="WalkH", save=True)

        for param in ["Stride", "Stance"]:
            boxplot_compare_algorithm_tsps("Diao", "TP-EAR",
                                       source, tsParam=param, save=True)





