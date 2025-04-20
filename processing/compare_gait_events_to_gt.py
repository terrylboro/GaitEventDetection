import os
from utils.data_manipulation_utils import lp_filter, find_nearest
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # to avoid loading errors with pandas
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("Qt5Agg")
import re


def convert_events_to_dict(filename, gtDF, imuDF, startIdx=None, endIdx=None, toleranceInterval=30):
    # Create dictionary
    gtDict = {}
    imuDict = {}

    if (not pd.isna(gtDF.loc[filename, "Left ICs"])) & (not pd.isna(imuDF.loc[filename, "Left ICs"])):

        # Firstly, figure out first IC and crop
        gtLICs = np.array([int(x) for x in re.sub(r'[\[\],]', '', gtDF.loc[filename, "Left ICs"]).strip().split(" ") if x != ""])
        gtRICs = np.array([int(x) for x in re.sub(r'[\[\],]', '', gtDF.loc[filename, "Right ICs"]).strip().split(" ") if x != ""])
        firstIC = min(np.concatenate((gtLICs, gtRICs)))
        lastIC = max(np.concatenate((gtLICs, gtRICs)))

        # Now add all the events to the dictionary
        for event in ["Left ICs", "Right ICs", "Left FOs", "Right FOs"]:
            # Skip if there's a null column
            if (not pd.isna(gtDF.loc[filename, event])) & (not pd.isna(imuDF.loc[filename, event])):
                gtEvents = np.array([int(x) for x in re.sub(r'[\[\],]', '', gtDF.loc[filename, event]).strip().split(" ") if x != ""])
                imuEvents = np.array([int(x) for x in re.sub(r'[\[\],]', '', imuDF.loc[filename, event]).strip().split(" ") if x != ""])

                # Crop out start/turn for TUG
                if startIdx is not None:
                    gtEvents = gtEvents[startIdx < gtEvents]
                    if "FOs" in event:
                        imuEvents = imuEvents[startIdx < imuEvents]
                    else:
                        imuEvents = imuEvents[startIdx + toleranceInterval < imuEvents]
                if endIdx is not None:
                    gtEvents = gtEvents[gtEvents < endIdx + 1]
                    if "FOs" in event:
                        imuEvents = imuEvents[imuEvents < endIdx]
                    else:
                        imuEvents = imuEvents[imuEvents < endIdx + toleranceInterval]

                # Crop out IMU-detected FOs which occur after last ground truth IC
                # We know these only after we've looped through ICs
                if event in ["Left FOs", "Right FOs"]:
                    imuEvents = imuEvents[(imuEvents > firstIC) & (imuEvents < lastIC)]
                    gtEvents = gtEvents[(gtEvents > firstIC) & (gtEvents < lastIC)]
                else:
                    imuEvents = imuEvents[
                        (imuEvents > firstIC - toleranceInterval) & (imuEvents < lastIC + toleranceInterval)]
                    gtEvents = gtEvents[
                        (gtEvents > firstIC - toleranceInterval) & (gtEvents < lastIC + toleranceInterval)]
                # Add to dictionary
                gtDict[event] = gtEvents.tolist()
                imuDict[event] = imuEvents.tolist()
            else:
                print("{} events are empty".format(filename))
                return {}, {}

    return gtDict, imuDict


def compare_gait_events_to_gt_with_laterality(gtArr, imuArr, toleranceInterval):
    missedIdxs = []
    extraIdxs = []
    lateralityErrors = []
    matchedICsList = []  # this list tracks matched ICs so that close steps aren't counted twice
    # Find the missed idxs
    diffs = np.full(gtArr.shape, np.nan)
    matchedEventIdxs = np.full((2, len(gtArr)), np.nan)
    matchedLateralityIdxs = np.full((2, len(gtArr)), np.nan, dtype=object)
    for i, gtIC in enumerate(gtArr):
        # Find the closest IMU events to GT
        matchedIC = imuArr[np.argmin(abs(imuArr["Event"] - gtIC["Event"]))]

        if (abs(gtIC["Event"] - matchedIC["Event"]) < toleranceInterval) and (matchedIC["Event"] not in matchedICsList):
            # print("Matched {} and {}!".format(gtIC["Event"], matchedIC["Event"]))
            # Log the matching events
            matchedEventIdxs[0, i] = gtIC["Event"]
            matchedEventIdxs[1, i] = matchedIC["Event"]
            matchedLateralityIdxs[0, i] = gtIC["Side"]
            matchedLateralityIdxs[1, i] = matchedIC["Side"]
            # Store this event to make sure it isn't repeated
            matchedICsList.append(matchedIC["Event"])
            # Find timing differences for given event
            diffs[i] = gtIC["Event"] - matchedIC["Event"]
            # Determine whether there was a laterality error
            # print("GT is {} and IMU is {}".format(gtIC["Side"], matchedIC["Side"]))
            if gtIC["Side"] != matchedIC["Side"]:
                lateralityErrors.append(gtIC["Event"])
        else:
            print("Missed event at {}".format(gtIC))
            missedIdxs.append(int(gtIC["Event"]))
            matchedEventIdxs[0, i] = gtIC["Event"]
            matchedLateralityIdxs[0, i] = gtIC["Side"]


    # Now find the extra idxs
    for i, IC in enumerate(imuArr):
        for gtIC in gtArr:
            if abs(gtIC["Event"] - IC["Event"]) < toleranceInterval:
                break
            elif gtIC == gtArr[-1]:
                print("Extra event found: {}".format(IC["Event"]))
                extraIdxs.append(int(IC["Event"]))

    missedIdxs = np.array(missedIdxs)
    extraIdxs = np.array(extraIdxs)
    lateralityErrors = np.array(lateralityErrors)
    return diffs, matchedEventIdxs, matchedLateralityIdxs, missedIdxs, extraIdxs, lateralityErrors


def compare_gait_events(gtEventsDict, imuEventsDict, event="ICs", toleranceInterval=30):
    """
    1. Detection of IC events regardless of laterality (missed/extra added)
    2. Detection of FO events regardless of laterality
    3. Laterality of IC and FO events
    :param gtDF:
    :param imuDF:
    :return:
    """
    # Step 3: Check laterality
    gtLICsTuples = [(x, "Left") for x in gtEventsDict["Left {}".format(event)]]
    gtRICsTuples = [(x, "Right") for x in gtEventsDict["Right {}".format(event)]]
    gtICsTuples = gtLICsTuples + gtRICsTuples
    gtICsTuples = sorted(gtICsTuples, key=lambda tup: tup[0])
    
    imuLICsTuples = [(x, "Left") for x in imuEventsDict["Left {}".format(event)]]
    imuRICsTuples = [(x, "Right") for x in imuEventsDict["Right {}".format(event)]]
    imuICsTuples = imuLICsTuples + imuRICsTuples
    imuICsTuples = sorted(imuICsTuples, key=lambda tup: tup[0])

    dtype = np.dtype([("Event", np.int32), ("Side", "U10")])
    
    diffs, matchedEventIdxs, matchedLateralityIdxs, missedIdxs, extraIdxs, lateralityErrors = compare_gait_events_to_gt_with_laterality(np.array(gtICsTuples, dtype=dtype), np.array(imuICsTuples, dtype=dtype), toleranceInterval)
    return diffs, matchedEventIdxs, matchedLateralityIdxs, missedIdxs, extraIdxs, lateralityErrors


def compare_tug_events(imuDF, gtDF):
    # Remove any dodgy trials
    imuDF = imuDF.dropna()
    gtDF = gtDF.drop(columns=["Notes"]).dropna()

    # ICs and FOs separately
    index = pd.Index(imuDF.index.values.tolist(), name="Filename")

    # Define col names
    # Structure to be Filename | Event Type (MultiIdx) | 20 x Diffs | Num Missed | Num extra | Laterality Error | Mean | Range
    colNames = ["Event" + str(x) for x in range(1, 25)]
    colNames.extend(["MissedIdxs", "ExtraIdxs"])
    ICdiffsDF = pd.DataFrame(index=index, data=np.full((len(index), 26), np.nan), columns=colNames)
    FOdiffsDF = pd.DataFrame(index=index, data=np.full((len(index), 26), np.nan), columns=colNames)
    # And build df to store both raw idxs
    events = np.tile(["GT ICs", "IMU ICs", "GT FOs", "IMU FOs"], len(imuDF)).tolist()
    diffsIndex = pd.MultiIndex.from_tuples(list(zip(*[np.repeat(imuDF.index.values, 4).tolist(), events])),
                                           names=["Filename", "EventType"])
    rawDiffsOverallDF = pd.DataFrame(index=diffsIndex, data=np.full((len(diffsIndex), 26), np.nan), columns=colNames)
    rawDiffsTurnsDF = pd.DataFrame(index=diffsIndex, data=np.full((len(diffsIndex), 26), np.nan), columns=colNames)
    rawDiffsStraightsDF = pd.DataFrame(index=diffsIndex, data=np.full((len(diffsIndex), 26), np.nan), columns=colNames)

    # Also define separate dfs for turning and non-turning parts to analyse separately
    colNames = ["Event" + str(x) for x in range(1, 20)]
    colNames.extend(["MissedIdxs", "ExtraIdxs"])
    ICturnsDF = pd.DataFrame(index=index, data=np.full((len(index), 21), np.nan), columns=colNames)
    FOturnsDF = pd.DataFrame(index=index, data=np.full((len(index), 21), np.nan), columns=colNames)
    ICstraightsDF = pd.DataFrame(index=index, data=np.full((len(index), 21), np.nan), columns=colNames)
    FOstraightsDF = pd.DataFrame(index=index, data=np.full((len(index), 21), np.nan), columns=colNames)

    # Load bad IMU trials to exclude
    badIMUTrials = pd.read_csv("../utils/info/badIMUtrials.csv", usecols=["Filename"]).to_numpy().flatten().tolist()
    # Load turns calculated from IMU
    turnsInfoDF = pd.read_csv("../utils/info/TUG Turning Events from IMU.csv", index_col="Filename")

    # Compare both
    for filename in imuDF.index.values:
        # Find the segmentation events
        # # If using the GT values:
        # firstStepIdx = int(gtDF.at[filename, "FirstStepIdx"])
        # endWalkIdx = int(gtDF.at[filename, "EndWalkIdx"])
        # turnsIdxs = gtDF.loc[filename, ["StartTurnIdx", "EndTurnIdx"]].astype(int).to_numpy()
        # Else if using the IMU values
        firstStepIdx = int(turnsInfoDF.at[filename, "FirstStepIdx"])
        endWalkIdx = int(turnsInfoDF.at[filename, "EndWalkIdx"])
        turnsIdxs = turnsInfoDF.loc[filename, ["StartTurnIdx", "EndTurnIdx"]].astype(int).to_numpy()

        subjectNum = int(filename[3:5])
        trialNum = int(filename[8:10])

        if 72 > subjectNum > 0 and (filename not in badIMUTrials):
            print(filename)
            # Firstly, handle the whole trial
            gtEventsDict, imuEventsDict = convert_events_to_dict(filename, gtDF, imuDF,
                                                                 startIdx=firstStepIdx, endIdx=endWalkIdx,
                                                                 toleranceInterval=30)
            print(gtEventsDict)
            print(imuEventsDict)
            print("First Step: ", firstStepIdx)
            print("Last Step: ", endWalkIdx)
            print("Turns area: {} to {}".format(turnsIdxs[0], turnsIdxs[1]))

            if (bool(gtEventsDict)) & (bool(imuEventsDict)):
                # Handle whole trial
                for event in ["ICs", "FOs"]:
                    diffs, matchedEventIdxs, matchedLateralityIdxs, missedIdxs, extraIdxs, lateralityErrors = compare_gait_events(gtEventsDict, imuEventsDict,
                                                                                         event=event, toleranceInterval=30)

                    # Log and save matched Event Idxs to file
                    rawDiffsOverallDF.loc[(filename, ["GT {}".format(event), "IMU {}".format(event)]), :matchedEventIdxs.shape[1]] = matchedEventIdxs
                    print(matchedEventIdxs[:, ((turnsIdxs[0] > matchedEventIdxs[0, :]) | (matchedEventIdxs[0, :] > turnsIdxs[1]))])
                    matchedStraightsIdxs = matchedEventIdxs[:, ((turnsIdxs[0] > matchedEventIdxs[0, :]) | (matchedEventIdxs[0, :] > turnsIdxs[1]))]
                    matchedTurnsIdxs = matchedEventIdxs[:, ((turnsIdxs[0] < matchedEventIdxs[0, :]) & (matchedEventIdxs[0, :] < turnsIdxs[1]))]

                    rawDiffsStraightsDF.loc[(filename, ["GT {}".format(event), "IMU {}".format(event)]), :matchedStraightsIdxs.shape[1]] = matchedStraightsIdxs
                    rawDiffsTurnsDF.loc[(filename, ["GT {}".format(event), "IMU {}".format(event)]), :matchedTurnsIdxs.shape[1]] = matchedTurnsIdxs
                    # Adjust for extraIdxs detected after final step
                    extraIdxs = np.array(extraIdxs)
                    extraIdxs = extraIdxs[(extraIdxs > firstStepIdx) & (extraIdxs < endWalkIdx-30)]
                    extraIdxs = extraIdxs.tolist()

                    turnsDiffs = np.diff(matchedEventIdxs[:, (matchedEventIdxs[0] > turnsIdxs[0]) & (matchedEventIdxs[0] < turnsIdxs[1])], axis=0).flatten()
                    nonTurnsDiffs = np.diff(matchedEventIdxs[:, (matchedEventIdxs[0] < turnsIdxs[0]) | (matchedEventIdxs[0] > turnsIdxs[1])], axis=0).flatten()

                    if event == "ICs":
                        ICdiffsDF.loc[filename, :"Event{:d}".format(len(diffs))] = diffs
                        ICdiffsDF.loc[filename, ["MissedIdxs", "ExtraIdxs", "LateralityErrors"]] = [len(missedIdxs),
                                                                                                       len(extraIdxs),
                                                                                                       len(lateralityErrors)]
                        ICturnsDF.loc[filename, :"Event{:d}".format(len(turnsDiffs))] = turnsDiffs
                        ICturnsDF.loc[filename, "MissedIdxs"] = 0 if len(missedIdxs) == 0 else \
                            ((turnsIdxs[0] < missedIdxs) & (missedIdxs < turnsIdxs[1])).sum()
                        ICturnsDF.loc[filename, "ExtraIdxs"] = 0 if len(extraIdxs) == 0 else \
                            ((turnsIdxs[0] < extraIdxs) & (extraIdxs < turnsIdxs[1])).sum()
                        ICturnsDF.loc[filename, "LateralityErrors"] = 0 if len(lateralityErrors) == 0 else \
                            ((turnsIdxs[0] < lateralityErrors) & (lateralityErrors < turnsIdxs[1])).sum()

                        ICstraightsDF.loc[filename, :"Event{:d}".format(len(nonTurnsDiffs))] = nonTurnsDiffs
                        ICstraightsDF.loc[filename, "MissedIdxs"] = 0 if len(missedIdxs) == 0 else \
                            ((turnsIdxs[0] > missedIdxs) | (missedIdxs > turnsIdxs[1])).sum()
                        ICstraightsDF.loc[filename, "ExtraIdxs"] = 0 if len(extraIdxs) == 0 else \
                            ((turnsIdxs[0] > extraIdxs) | (extraIdxs > turnsIdxs[1])).sum()
                        ICstraightsDF.loc[filename, "LateralityErrors"] = 0 if len(lateralityErrors) == 0 else \
                            ((turnsIdxs[0] > lateralityErrors) | (lateralityErrors > turnsIdxs[1])).sum()

                    else:
                        FOdiffsDF.loc[filename, :"Event{:d}".format(len(diffs))] = diffs
                        FOdiffsDF.loc[filename, ["MissedIdxs", "ExtraIdxs", "LateralityErrors"]] = [len(missedIdxs),
                                                                                                       len(extraIdxs),
                                                                                                       len(lateralityErrors)]
                        FOturnsDF.loc[filename, :"Event{:d}".format(len(turnsDiffs))] = turnsDiffs
                        FOturnsDF.loc[filename, "MissedIdxs"] = 0 if len(missedIdxs) == 0 else \
                            ((turnsIdxs[0] < missedIdxs) & (missedIdxs < turnsIdxs[1])).sum()
                        FOturnsDF.loc[filename, "ExtraIdxs"] = 0 if len(extraIdxs) == 0 else \
                            ((turnsIdxs[0] < extraIdxs) & (extraIdxs < turnsIdxs[1])).sum()
                        FOturnsDF.loc[filename, "LateralityErrors"] = 0 if len(lateralityErrors) == 0 else \
                            ((turnsIdxs[0] < lateralityErrors) & (lateralityErrors < turnsIdxs[1])).sum()

                        FOstraightsDF.loc[filename, :"Event{:d}".format(len(nonTurnsDiffs))] = nonTurnsDiffs
                        FOstraightsDF.loc[filename, "MissedIdxs"] = 0 if len(missedIdxs) == 0 else \
                            ((turnsIdxs[0] > missedIdxs) | (missedIdxs > turnsIdxs[1])).sum()
                        FOstraightsDF.loc[filename, "ExtraIdxs"] = 0 if len(extraIdxs) == 0 else \
                            ((turnsIdxs[0] > extraIdxs) | (extraIdxs > turnsIdxs[1])).sum()
                        FOstraightsDF.loc[filename, "LateralityErrors"] = 0 if len(lateralityErrors) == 0 else \
                            ((turnsIdxs[0] > lateralityErrors) | (lateralityErrors > turnsIdxs[1])).sum()
                    print("Missed events: ", missedIdxs)
                    print("Laterality Errors: ", lateralityErrors)
                    print("Extra Events: ", extraIdxs)

    return ICdiffsDF, FOdiffsDF, ICturnsDF, FOturnsDF, ICstraightsDF, FOstraightsDF, rawDiffsOverallDF, rawDiffsTurnsDF, rawDiffsStraightsDF


def compare_walking_events(imuDF, gtDF):
    # ICs and FOs together
    # filenames = np.repeat(imuDF.index.values, 2).tolist()
    # events = np.tile(["ICs", "FOs"], len(imuDF)).tolist()
    # index = pd.MultiIndex.from_tuples(list(zip(*[filenames, events])), names=["Filename", "EventType"])

    # ICs and FOs separate
    index = pd.Index(imuDF.index.values.tolist(), name="Filename")

    # Define col names
    colNames = ["Event" + str(x) for x in range(1, 25)]
    colNames.extend(["MissedIdxs", "ExtraIdxs"])
    ICdiffsDF = pd.DataFrame(index=index, data=np.full((len(index), 26), np.nan), columns=colNames)
    FOdiffsDF = pd.DataFrame(index=index, data=np.full((len(index), 26), np.nan), columns=colNames)

    # And build df to store both raw idxs
    events = np.tile(["GT ICs", "IMU ICs", "GT FOs", "IMU FOs"], len(imuDF)).tolist()
    diffsIndex = pd.MultiIndex.from_tuples(list(zip(*[np.repeat(imuDF.index.values, 4).tolist(), events])),
                                           names=["Filename", "EventType"])
    rawDiffsOverallDF = pd.DataFrame(index=diffsIndex, data=np.full((len(diffsIndex), 26), np.nan), columns=colNames)


    # Load bad IMU trials to exclude
    badIMUTrials = pd.read_csv("../utils/info/badIMUtrials.csv", usecols=["Filename"]).to_numpy().flatten().tolist()
    print("Bad IMU Trials: ", badIMUTrials)
    # Compare both
    for filename in imuDF.index.values:
        subjectNum = int(filename[3:5])
        trialNum = int(filename[8:10])
        if subjectNum > 0 and filename not in badIMUTrials:
            # try:
            print(filename)
            # Test new function
            gtEventsDict, imuEventsDict = convert_events_to_dict(filename, gtDF, imuDF)
            print(gtEventsDict)
            print(imuEventsDict)
            if (bool(gtEventsDict)) & (bool(imuEventsDict)):

                for event in ["ICs", "FOs"]:
                    diffs, matchedEventIdxs, matchedLateralityIdxs, missedIdxs, extraIdxs, lateralityErrors = compare_gait_events(gtEventsDict, imuEventsDict,
                                                                                         event=event, toleranceInterval=30)

                    # Log and save matched Event Idxs to file
                    rawDiffsOverallDF.loc[(filename, ["GT {}".format(event), "IMU {}".format(event)]),
                    :matchedEventIdxs.shape[1]] = matchedEventIdxs

                    if event == "ICs":
                        ICdiffsDF.loc[filename, :"Event{:d}".format(len(diffs))] = diffs
                        ICdiffsDF.loc[filename, ["MissedIdxs", "ExtraIdxs", "LateralityErrors"]] = [len(missedIdxs),
                                                                                                    len(extraIdxs),
                                                                                                    len(lateralityErrors)]
                    else:
                        FOdiffsDF.loc[filename, :"Event{:d}".format(len(diffs))] = diffs
                        FOdiffsDF.loc[filename, ["MissedIdxs", "ExtraIdxs", "LateralityErrors"]] = [len(missedIdxs),
                                                                                                    len(extraIdxs),
                                                                                                    len(lateralityErrors)]

    # # Add mean, range, std
    # diffsDF["MAE"] = diffsDF.loc[:, "Event1":"Event19"].abs().mean(skipna=True, axis=1)
    # diffsDF["Range"] = diffsDF.loc[:, "Event1":"Event19"].max(skipna=True, axis=1) - diffsDF.loc[:, "Event1":"Event19"].min(skipna=True, axis=1)
    return ICdiffsDF, FOdiffsDF, rawDiffsOverallDF


if __name__ == "__main__":
    # activity = "TUG"
    # algorithm = "FFT"  # Original or FFT

    for algorithm in ["TP-EAR"]:  # ,
        for activity in ["WalkShake"]:

            # Open both the events DF
            gtDF = pd.read_csv("../utils/Ground Truth Events/{} Gait Events.csv".format(activity), index_col="Filename")

            imuDF = pd.read_csv("../data/{}/Events/{}Events.csv".format(algorithm, activity), index_col="Filename")
            imuDF = imuDF.dropna()

            if activity == "TUG":
                ICdiffs, FOdiffs, turnsICDiffs, turnsFODiffs, straightsICDiffs, straightsFODiffs, rawDiffsOverallDF, rawDiffsTurnsDF, rawDiffsStraightsDF = compare_tug_events(imuDF, gtDF)
                turnsICDiffs.to_csv("../data/{}/Diffs/TUGTurnsDiffsIC.csv".format(algorithm))
                turnsFODiffs.to_csv("../data/{}/Diffs/TUGTurnsDiffsFO.csv".format(algorithm))
                straightsICDiffs.to_csv("../data/{}/Diffs/TUGNonTurnsDiffsIC.csv".format(algorithm))
                straightsFODiffs.to_csv("../data/{}/Diffs/TUGNonTurnsDiffsFO.csv".format(algorithm))
                rawDiffsTurnsDF.to_csv("../data/{}/RawDiffs/TUGTurnsDiffsRaw.csv".format(algorithm))
                rawDiffsStraightsDF.to_csv("../data/{}/RawDiffs/TUGNonTurnsDiffsRaw.csv".format(algorithm))
            else:
                ICdiffs, FOdiffs, rawDiffsOverallDF = compare_walking_events(imuDF, gtDF)

            ICdiffs.to_csv("../data/{}/Diffs/{}DiffsIC.csv".format(algorithm, activity))
            FOdiffs.to_csv("../data/{}/Diffs/{}DiffsFO.csv".format(algorithm, activity))
            rawDiffsOverallDF.to_csv("../data/{}/RawDiffs/{}DiffsRawOverall.csv".format(algorithm, activity))

