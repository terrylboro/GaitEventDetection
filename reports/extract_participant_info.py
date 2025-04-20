import os
import pandas as pd
import numpy as np
from utils.participant_info_utils import get_non_typical_participant_nums

# Load participant info
infoDF = pd.read_csv("../utils/info/ParticipantInfo.csv", index_col="Participant")
infoDF = infoDF.iloc[:, 0:6]  # get rid of blank columns
infoDF["Sex"] = infoDF["Sex"].astype("category")
# print(infoDF.head())

nonTypicalNums = get_non_typical_participant_nums()
typicalNums = [x for x in range(1, 72) if x not in nonTypicalNums]
nonTypicalDF = infoDF.loc[nonTypicalNums, :]
nonTypicalDF["Reason"] = nonTypicalDF["Reason"].astype("category")
typicalDF = infoDF.loc[typicalNums, :]

def calculate_plus_minus(df):
    return max(abs(df.mean() - df.min()), abs(df.mean() - df.max()))


def calculate_iqr(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    print("Max {} 25 {} 75 {}".format(df.max(), Q1, Q3))
    return Q3 - Q1


summaryDF = pd.DataFrame(columns=["Characteristic", "Typical", "Non-Typical"], dtype=object)
summaryDF = summaryDF.set_index("Characteristic")
for df, name in zip([typicalDF, nonTypicalDF], ["Typical", "Non-Typical"]):
    print(dict(df.Sex.value_counts()))
    # summaryDF.loc["Sex (M/F)", name] = "{}/{}".format(dict(df.Sex.value_counts())["M"], dict(df.Sex.value_counts())["F"])
    # summaryDF.loc["Age (Years)", name] = "{:2.1f} ± {:2.1f}".format(df.Age.mean(),
    #                                                             calculate_plus_minus(df.Age))
    # summaryDF.loc["Height (m)", name] = "{:2.1f} ± {:2.1f}".format(df.Height.mean(),
    #                                                             calculate_plus_minus(df.Height))
    # summaryDF.loc["Bodyweight (kg)", name] = "{:2.1f} ± {:2.1f}".format(df.Weight.mean(),
    #                                                             calculate_plus_minus(df.Weight))

    summaryDF.loc["Sex (M/F)", name] = "{}/{}".format(dict(df.Sex.value_counts())["M"],
                                                      dict(df.Sex.value_counts())["F"])
    summaryDF.loc["Age (Years)", name] = "{:2.1f} ({:2.1f})".format(df.Age.median(),
                                                                    calculate_iqr(df.Age))
    summaryDF.loc["Height (m)", name] = "{:2.1f} ({:2.1f})".format(df.Height.median(),
                                                                   calculate_iqr(df.Height))
    summaryDF.loc["Bodyweight (kg)", name] = "{:2.1f} ({:2.1f})".format(df.Weight.median(),
                                                                        calculate_iqr(df.Weight))



print(summaryDF)
# summaryDF.to_csv("Participant Summary.csv")
# print(nonTypicalDF.Reason.value_counts())

# Age
# Height
# Weight
