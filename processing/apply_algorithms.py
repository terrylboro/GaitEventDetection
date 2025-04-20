from utils.orientation_utils import calculate_tilt_corrected_linear_acceleration_adapted
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from utils.participant_info_utils import get_generic_trial_nums

mpl.use("Qt5Agg")

from adaptedDiao import apply_adapted_diao
from TPEAR import apply_tp_ear


def process_walking_trials(activityName, algorithm, selectFreq):
    calculate_events = apply_adapted_diao if algorithm == "Diao" else apply_tp_ear
    offsetDF = pd.read_csv("../utils/info/IMU Offsets.csv", index_col="Filename")
    for subjectNum in [x for x in range(60, 61) if x not in [46, 47, 48]]:
        for trialNum in [13]:#get_generic_trial_nums(subjectNum, [activityName]):
            opticalFilename = "TF_{:02d}_{:04d}.c3d".format(subjectNum, trialNum)
            print(opticalFilename)
            if subjectNum in [1, 6]:
                trialNum += 1
            imuFilename = "TF_{:02d}-{:02d}.csv".format(subjectNum, trialNum)
            try:
                imuDF = pd.read_csv(os.path.join("../imu/data/TF_{:02d}/".format(subjectNum), imuFilename))
                # Apply the offset
                offset = int(offsetDF.loc[opticalFilename, "Avg"])
                imuDF = imuDF.loc[offset:]
                imuDF = imuDF.reset_index(drop=True)
                # Resolve XYZ directions to fit world frame
                linAccResolved = calculate_tilt_corrected_linear_acceleration_adapted(imuDF, gVal=9.81)
                for side in ["lear"]:
                    data = linAccResolved[:, [1, 2]]
                    try:
                        LICs_l, RICs_l, LTCs_l, RTCs_l, fig, axs = calculate_events(data, 200, side, selectFreq, False)
                    except:
                        LICs_l, RICs_l, LTCs_l, RTCs_l, fig, axs = calculate_events(data, 100, side, selectFreq,
                                                                                      False)
                    # add these to df
                    if selectFreq:
                        eventsDFPath = "../data/{}/Events/{}Events.csv".format(algorithm, activityName)
                    else:
                        eventsDFPath = "../data/{}/Events/{}DiaoEventsOriginal.csv".format(algorithm, activityName)
                    eventsDF = pd.read_csv(eventsDFPath, index_col="Filename",
                                               dtype="object")

                    eventsDF.loc[opticalFilename, "Left ICs"] = LICs_l
                    eventsDF.loc[opticalFilename, "Right ICs"] = RICs_l
                    eventsDF.loc[opticalFilename, "Left FOs"] = LTCs_l
                    eventsDF.loc[opticalFilename, "Right FOs"] = RTCs_l

                    print("Saving to {}".format(eventsDFPath))
                    eventsDF.to_csv(eventsDFPath)
                    # ##################

                    # fig.suptitle("{} - {}".format(imuFilename.split(".")[0], side), fontsize=20)
                    # fig.tight_layout()
                    # fig.subplots_adjust(top=0.88)

                    # plt.show()
                    # plt.savefig("../plots/{}/{}.png".format(activityName, opticalFilename.split(".")[0]))
                    # plt.close()
            except Exception as e:
                print(e)


def process_tug_trials(algorithm, selectFreq=True):
    calculate_events = apply_adapted_diao if algorithm == "Diao" else apply_tp_ear
    tugEventsDF = pd.read_csv("../utils/Ground Truth Events/TUG Gait Events.csv", index_col="Filename")
    infoDF = pd.read_csv("../utils/info/Sit2WalkInfo.csv", index_col="Filename")
    filelist = tugEventsDF.index.tolist()
    for opticalFilename in filelist:
        if infoDF.at[opticalFilename, "IMUDataGood"] == "Good":
            # Retrieve files for selected activity and participant
            subjectNum = int(opticalFilename[3:5])
            trialNum = int(opticalFilename[8:10])
            if subjectNum in [1, 6]:
                trialNum += 1
            imuFilename = "TF_{:02d}-{:02d}.csv".format(subjectNum, trialNum)
            offset = infoDF.at[opticalFilename, "Offset"]
            firstStepIdx = infoDF.at[opticalFilename, "FirstStepIdx"]
            # Find turns for plotting and crop to first step
            turnsIdxs = tugEventsDF.loc[opticalFilename, ["StartTurnIdx", "EndTurnIdx", "EndWalkIdx"]]
            turnsIdxs -= firstStepIdx
            # Apply correction to ensure first step is included
            firstStepIdx -= 3
            if (72 > subjectNum > 0) and (subjectNum not in [46, 47, 48]):

                # if (subjectNum == 52 and trialNum == 24) and (subjectNum not in [46, 47, 48]):
                print(imuFilename)
                imuDF = pd.read_csv(os.path.join("../imu/data/TF_{:02d}".format(subjectNum), imuFilename))
                # linAccResolved = calculate_tilt_corrected_linear_acceleration_adapted(imuDF, gVal=9.81)
                imuDF = imuDF.loc[firstStepIdx + offset:tugEventsDF.at[
                    opticalFilename, "EndWalkIdx"] + offset + 5]  # Align IMU and optical systems
                for sideNum, side in enumerate(["lear"]):
                    data = imuDF[["AccY" + side, "AccZ" + side]].to_numpy()
                    LICs_l, RICs_l, LTCs_l, RTCs_l, fig, axs = calculate_events(data, 100, side, selectFreq, False)

                    # add these to df
                    if selectFreq:
                        eventsDFPath = "../data/{}/Events/TUGEvents.csv".format(algorithm)
                    else:
                        eventsDFPath = "../data/{}/Events/TUGDiaoEventsOriginal.csv".format(algorithm)

                    diaoEventsDF = pd.read_csv(eventsDFPath, index_col="Filename", dtype="object")
                    diaoEventsDF.loc[opticalFilename, "Left ICs"] = LICs_l + firstStepIdx
                    diaoEventsDF.loc[opticalFilename, "Right ICs"] = RICs_l + firstStepIdx
                    diaoEventsDF.loc[opticalFilename, "Left FOs"] = LTCs_l + firstStepIdx
                    diaoEventsDF.loc[opticalFilename, "Right FOs"] = RTCs_l + firstStepIdx
                    diaoEventsDF.to_csv(eventsDFPath)
                    ##################

                    # if turnsIdxs is not None:
                    #     axs[0, 0].vlines(turnsIdxs[:-1], axs[0, 0].get_ylim()[0], axs[0, 0].get_ylim()[1], color='y', linestyle='-.')
                    #     axs[0, 1].vlines(turnsIdxs[:-1], axs[0, 1].get_ylim()[0], axs[0, 1].get_ylim()[1], color='y', linestyle='-.')
                    #     axs[1, 0].vlines(turnsIdxs[:-1], axs[1, 0].get_ylim()[0], axs[1, 0].get_ylim()[1], color='y', linestyle='-.')
                    #     axs[1, 1].vlines(turnsIdxs[:-1], axs[1, 1].get_ylim()[0], axs[1, 1].get_ylim()[1], color='y', linestyle='-.')
                    #
                    # # Adjust x tick values
                    # for i in [0, 1]:
                    #     for j in [0, 1]:
                    #         currentLabels = axs[i, j].get_xticks()
                    #         axs[i, j].set_xticklabels([int(x) for x in currentLabels + int(firstStepIdx + int(offset))])


                    # fig.suptitle("{} - {}".format(opticalFilename.split(".")[0], side), fontsize=20)
                    # fig.tight_layout()
                    # fig.subplots_adjust(top=0.88)
                    # plt.savefig("../plots/{}/{}.png".format("TUG/Diao/", opticalFilename.split(".")[0]))

                    # plt.show()
                    # plt.close()


if __name__ == "__main__":
    selectFreq = True  # Ensures correct SSA component is selected
    for algorithm in ["Diao"]:  #
        for activity in ["WalkShake"]:
            process_walking_trials(activity, algorithm, selectFreq)
        # process_tug_trials(algorithm)
