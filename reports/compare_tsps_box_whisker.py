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
        tspsDF["{}Diffs".format(tspName)] = (tspsDF["GT{}Time".format(tspName)] - tspsDF[
            "IMU{}Time".format(tspName)])

    return tspsDF


def boxplot_separate_tsps_by_state(tspsDF, plotName, dataSource, save=False, isNew=False):
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
        raise("Unrecognized data source: {}".format(dataSource))

    for tspName in ["Stride", "Stance", "Swing"]:
        # Plot TSP by condition
        ax = sns.boxplot(x=tspsDF["State"], y=tspsDF[eventStr.format(tspName)] / 100, hue=tspsDF["State"],
                         hue_order=["Typical", "Non-Typical"], order=["Typical", "Non-Typical"])

        plt.xlabel("Condition")
        plt.ylabel("Time / s")
        plt.title("{} - {} ".format(plotName, tspName) + title)
        if save:
            if isNew:
                plt.savefig("NewAlgorithm/{}-{}-{}-TypicalvsNonTypical.png".format(plotName, tspName, dataSource))
                plt.close()
            else:
                plt.savefig("BoxWhisker/{}-{}-{}-TypicalvsNonTypical.png".format(plotName, tspName, dataSource))
                plt.close()
        else:
            plt.show()


def boxplot_all_tsps_by_state(tspsDF, plotName, dataSource, save=False, isNew=False):
    """
    Create a boxplot for stride, stance and swing on a single plot.
    Typical vs non-typical participants are plotted for each event.
    :param tspsDF:
    :param plotName:
    :param dataSource:
    :param title:
    :param save:
    :return:
    """
    if dataSource == "IMU":
        eventStr = "IMU{}Time"
        title = "{} TSPs Calculated with IMUs".format(plotName)
    elif dataSource == "GT":
        eventStr = "GT{}Time"
        title = "{} Ground Truth TSPs".format(plotName)
    elif dataSource == "Diffs":
        eventStr = "{}Diffs"
        title = "{} TSP Difference between IMU and Ground Truth".format(plotName)
    else:
        raise("Unrecognized data source: {}".format(dataSource))

    # Reformat the df to fit boxplot format
    boxDF = pd.DataFrame(columns=["State", "EventType", "Value"])
    for cycle in tspsDF.index.values:
        state = tspsDF.at[cycle, "State"]
        eventArr = np.full((3, 3), fill_value=np.nan, dtype="object")
        for i, eventName in enumerate(["Stride", "Stance", "Swing"]):
            eventArr[i] = [state, eventName, tspsDF.at[cycle, eventStr.format(eventName)] / 100]
        # Add to overall df
        boxDF = pd.concat((boxDF, pd.DataFrame(data=eventArr, columns=["State", "EventType", "Value"])))
    # Ensure the types are categorical and indexed correctly then plot
    boxDF = boxDF.astype(dtype={"State": "category", "EventType": "category", "Value": "float"})
    boxDF.State = pd.Categorical(boxDF.State, categories=["Typical", "Non-Typical"], ordered=True)
    boxDF.EventType = pd.Categorical(boxDF.EventType, categories=["Stride", "Stance", "Swing"], ordered=True)
    boxDF.Value = boxDF.Value.astype(float)
    boxDF = boxDF.reset_index()
    sns.boxplot(x="EventType", y="Value", hue="State", data=boxDF)
    plt.title(title)
    plt.xlabel("Parameter")
    plt.ylabel("Time / s")

    if save:
        if isNew:
            plt.savefig("NewAlgorithm/{}-AllTSPs-TypicalvsNonTypical-{}.png".format(plotName, dataSource))
            plt.close()
        else:
            plt.savefig("BoxWhisker/{}-AllTSPs-TypicalvsNonTypical-{}.png".format(plotName, dataSource))
            plt.close()
    else:
        plt.show()


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

    for state in ["Typical", "Non-Typical"]:
        # Reformat the df to fit boxplot format
        boxDF = pd.DataFrame(columns=["Activity", "EventType", "Value"])
        for activityTitle, activityName in zip(["Walk", "WalkNod", "WalkShake", "TUG Turns"], ["Walk", "WalkV", "WalkH", "Turn"]):
            tspsDF1 = format_tsps_df(activityTitle, isNew=True)
            tspsDF2 = format_tsps_df(activityTitle, isNew=False)
            for tspsDF, algorithmName in zip([tspsDF1.loc[tspsDF1.State==state, :], tspsDF2.loc[tspsDF2.State==state, :]], [plotName1, plotName2]):
                for cycle in tspsDF.index.values:
                    eventArr = np.full((1, 4), fill_value=np.nan, dtype="object")
                    for i, eventName in enumerate([tsParam]):
                        eventArr[i] = [activityName, algorithmName, eventName, tspsDF.at[cycle, eventStr.format(eventName)] / 100]
                    # Add to overall df
                    boxDF = pd.concat((boxDF, pd.DataFrame(data=eventArr, columns=["Activity", "Algorithm", "EventType", "Value"])))

        # Ensure the types are categorical and indexed correctly then plot
        boxDF = boxDF.astype(dtype={"Activity": "category", "Algorithm": "category", "EventType": "category", "Value": "float"})
        boxDF.Activity = pd.Categorical(boxDF.Activity, categories=["Walk", "WalkV", "WalkH", "Turn"], ordered=True)
        boxDF.Algorithm = pd.Categorical(boxDF.Algorithm, categories=["Diao", "New"])
        boxDF.EventType = pd.Categorical(boxDF.EventType, categories=["Stance"], ordered=True)
        boxDF.Value = boxDF.Value.astype(float)
        boxDF = boxDF.reset_index()
        sns.boxplot(x="Activity", y="Value", hue="Algorithm", data=boxDF)
        plt.title(title + "\n{} Participants".format(state))
        plt.xlabel("Parameter")
        plt.ylabel("Time / s")

        if save:
            # plt.savefig("BoxWhisker/{}.png".format("-".join(str(title + "{} Participants".format(state)).split(" "))))
            plt.savefig("NewAlgorithm/CompareAlgorithms/{}.png".format("-".join(str(title + "{} Participants".format(state)).split(" "))))
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

        boxplot_compare_algorithm_tsps("New", "Diao",
                                   source, tsParam="Stride", save=True)





