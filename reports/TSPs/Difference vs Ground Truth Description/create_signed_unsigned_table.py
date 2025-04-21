import os
import pandas as pd


def convert_to_latex(table):
    for row in table.index[:3]:
        if "Stride" in row:
            rowStr = "\multirow{ 3}{*}{AE} & " + str(row) + " & " + " & ".join(table.loc[row, :]) + "\\\\"
        else:
            rowStr = "& " + str(row) + " & " + " & ".join(table.loc[row, :]) + "\\\\"
        rowStr = rowStr.replace("±", "\pm")
        rowStr = rowStr.replace("US", "")
        print(rowStr)
    print("\midrule")
    for row in table.index[3:]:
        if "Stride" in row:
            rowStr = "\multirow{ 3}{*}{SE} & " + str(row) + " & " + " & ".join(table.loc[row, :]) + "\\\\"
        else:
            rowStr = "& " + str(row) + " & " + " & ".join(table.loc[row, :]) + "\\\\"
        rowStr = rowStr.replace("±", "\pm")
        rowStr = rowStr.replace("S ", " ")
        print(rowStr)
    print("\\bottomrule")

    # return table


columns = pd.MultiIndex.from_tuples([('Walk', 'Diao'), ('Walk', 'TP-EAR'),
                                         ('WalkNod', 'Diao'), ('WalkNod', 'TP-EAR'),
                                         ('WalkShake', 'Diao'), ('WalkShake', 'TP-EAR'),
                                         ('TUG Turns', 'Diao'), ('TUG Turns', 'TP-EAR')])

typicalTable = pd.DataFrame(columns=columns,
                            index=pd.Index(["StrideUS", "StanceUS", "SwingUS",
                                            "StrideS", "StanceS", "SwingS"]))
nonTypicalTable = pd.DataFrame(columns=columns,
                            index=pd.Index(["StrideUS", "StanceUS", "SwingUS",
                                            "StrideS", "StanceS", "SwingS"]))


for filename in os.listdir(os.getcwd()):
    if filename.endswith('.csv'):
        df = pd.read_csv(filename, usecols=["Activity", "Algorithm", "Mean", "Std"])
        # Create formatted vals for df
        valuesList = []
        for row in range(len(df)):
            valuesList.append("${:2.0f}±{:2.0f}$".format(df.loc[row, "Mean"], df.loc[row, "Std"]))
        for activity in ["Stride", "Stance", "Swing"]:
            if activity in filename:
                if "Non-Typical" in filename:
                    if "Signed" in filename:
                        nonTypicalTable.loc[activity+"S", :] = valuesList#df.Mean.to_numpy()
                    else:
                        nonTypicalTable.loc[activity+"US", :] = valuesList#df.Mean.to_numpy()
                else:
                    if "Signed" in filename:
                        typicalTable.loc[activity+"S", :] = valuesList#df.Mean.to_numpy()
                    else:
                        typicalTable.loc[activity+"US", :] = valuesList#df.Mean.to_numpy()

# print(typicalTable)
# for row in typicalTable.index:
#     print(" & ".join(typicalTable.loc[row, :]))
# print(nonTypicalTable)

print("--- Typical ---\n")
convert_to_latex(typicalTable)
print("\n--- Non Typical ---\n")
convert_to_latex(nonTypicalTable)

# for activityName in ["Walk", "WalkShake", "WalkNod", "TUG Turns", "TUG Non Turns"]:
#     for algorithm in ["Diao", "TP-EAR"]:
#         print(activityName + "\n------------")
#         typicalDF, nonTypicalDF = format_tsps_df(activityName, algorithm)
#         print(typicalDF)
#         typicalTable[(activityName, algorithm)], nonTypicalTable[(activityName, algorithm)] = calculate_tsp_metrics_by_state(typicalDF, nonTypicalDF, "Diffs")