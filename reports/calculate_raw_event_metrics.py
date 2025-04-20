import pandas as pd
import os
from utils.participant_info_utils import get_non_typical_participant_nums
import re

# Make a template table to hold the data
# table = pd.DataFrame(columns=['Event', 'Walk', 'WalkH', 'WalkV', 'TUGTurn', 'TUGStraight'])
# table.loc[:, 'Event'] = ['IC', 'FO', 'Laterality']
# table.to_csv("Sensitivity Table.csv", index=False)

# New table which separates Typical and Non-Typical participants
columns = pd.MultiIndex.from_tuples([('Walk', 'Diao'), ('Walk', 'TP-EAR'),
                                         ('WalkNod', 'Diao'), ('WalkNod', 'TP-EAR'),
                                         ('WalkShake', 'Diao'), ('WalkShake', 'TP-EAR'),
                                         ('TUGTurns', 'Diao'), ('TUGTurns', 'TP-EAR'),
                                         ('TUGNonTurns', 'Diao'), ('TUGNonTurns', 'TP-EAR')])

typicalTable = pd.DataFrame(columns=columns, index=pd.Index(data=['IC', 'FO', 'Laterality'], name='Event'))
nonTypicalTable = pd.DataFrame(columns=columns, index=pd.Index(data=['IC', 'FO', 'Laterality'], name='Event'))
print(pd.Index(data=['IC', 'FO', 'Laterality'], name='Event'))
# print(table)

# We want to test all the files in diffsSplit
# fileDir = "../TUG/diffsSplit/"
# fileDir = "../TUG/New Data/diffsSplit/"
for algorithm in ["Diao", "TP-EAR"]:
    fileDir = "../data/{}/Diffs/".format(algorithm)
    for filename in os.listdir(fileDir):
        # load the df
        df = pd.read_csv(os.path.join(fileDir, filename), index_col=["Filename"])

        # Firstly, count # of events in total
        diffsDF = df.filter(regex=r"Event")

        # Process all the results together
        numDetectedEvents = diffsDF.count().sum()
        numMissedEvents = df["MissedIdxs"].sum()
        numLateralityErrors = df["LateralityErrors"].sum()
        sensitivity = numDetectedEvents / (numMissedEvents + numDetectedEvents)
        laterality = (numDetectedEvents - numLateralityErrors) / numDetectedEvents
        # print("{} - Sensitivity: {:2.1f}".format(filename, sensitivity * 100))
        # print("{} - Laterality: {:2.1f}".format(filename, laterality * 100))


        # Separate typical and non-typical results
        nonTypicalNums = get_non_typical_participant_nums()
        typicalNums = [x for x in range(1, 72) if x not in nonTypicalNums]
        nonTypicalMask = []
        for indexName in df.index.values:
            if int(indexName[3:5]) in nonTypicalNums:
                nonTypicalMask.append(True)
            else:
                nonTypicalMask.append(False)

        colName = re.sub(r'DiffsIC.csv', '', filename, flags=re.IGNORECASE)
        colName = re.sub(r'DiffsFO.csv', '', colName, flags=re.IGNORECASE)
        print(colName)

        event = "IC" if "IC" in filename else "FO"

        for state in ["Typical", "Non-Typical"]:
            mask = nonTypicalMask if state == "Non-Typical" else [not(x) for x in nonTypicalMask]
            maskedDF = df.loc[mask, :]
            maskedDiffsDF = diffsDF.loc[mask, :]
            numDetectedEvents = maskedDiffsDF.count().sum()
            numMissedEvents = maskedDF["MissedIdxs"].sum()
            numLateralityErrors = maskedDF["LateralityErrors"].sum()
            print("{} {} {}: Missed: {} / {}".format(colName, algorithm, state, numMissedEvents, (numMissedEvents + numDetectedEvents)))
            sensitivity = numDetectedEvents / (numMissedEvents + numDetectedEvents)
            laterality = (numDetectedEvents - numLateralityErrors) / numDetectedEvents
            print("{} - {} - Sensitivity: {:2.1f}".format(filename, state, sensitivity * 100))
            print("{} - {} - Laterality: {:2.1f}".format(filename, state, laterality * 100))

            # Add to table
            if state == "Typical":
                typicalTable.loc[event, (colName, algorithm)] = (sensitivity * 100).round(1)
                if event == "IC":
                    typicalTable.loc["Laterality", (colName, algorithm)] = (laterality * 100).round(1)
            else:
                nonTypicalTable.loc[event, (colName, algorithm)] = (sensitivity * 100).round(1)
                if event == "IC":
                    nonTypicalTable.loc["Laterality", (colName, algorithm)] = (laterality * 100).round(1)


# Drop TUG Non turns
typicalTable = typicalTable.drop(columns=["TUGNonTurns", "TUG"])
nonTypicalTable = nonTypicalTable.drop(columns=["TUGNonTurns", "TUG"])
print("----- Typical -----")
print(typicalTable)
print("----- Non Typical -----")
print(nonTypicalTable)

typicalTable.to_csv("Sensitivity/Sensitivity Table by Algorithm (Typical).csv")
nonTypicalTable.to_csv("Sensitivity/Sensitivity Table by Algorithm (Non Typical).csv")
