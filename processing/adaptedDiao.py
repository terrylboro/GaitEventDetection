from utils.gait_utils import *
from utils.orientation_utils import calculate_tilt_corrected_linear_acceleration_adapted
from utils.participant_info_utils import get_generic_trial_nums
from processing.ssa import SSA
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from utils.data_manipulation_utils import lp_filter
import matplotlib as mpl
mpl.use("Qt5Agg")


def detect_ic_adapted(acc_si_dominant, acc_ml_dominant, sample_rate_hz: float):
    """
    See https://www.mdpi.com/1424-8220/23/14/6565#sec2dot2-sensors-23-06565 .
    :param acc_si_dominant:
    :param acc_ml_dominant:
    :param sample_rate_hz:
    :return:
    """
    from scipy.signal import find_peaks
    import numpy as np
    # find minimum on SI axis with removed trend. Peaks corresponds to IC
    peaks_ic, _ = find_peaks(acc_si_dominant, distance=sample_rate_hz * 0.2)

    # determine ipsilateral and contralateral IC
    sides = []
    for peakDiff in (acc_ml_dominant.diff().loc[peaks_ic + 1]):
        if peakDiff > 0:
            sides.append("contralateral")
        else:
            sides.append("ipsilateral")
    return peaks_ic, np.array(sides)


def detect_tc(acc_ml_wo_trend, ic, sides):
    import numpy as np
    from scipy.signal import argrelmin, argrelmax
    mins = argrelmin(acc_ml_wo_trend.to_numpy(), order=2)[0]  # changes np.greater used to be np.less
    maxs = argrelmax(acc_ml_wo_trend.to_numpy(), order=2)[0]  # changes  np.less used to be np.greater

    # Find ipsilateral FOs
    contra = ic[sides == "contralateral"]
    contra = contra[contra < maxs[-1]]
    tc_ipsi = [int(maxs[maxs > x][0]) for x in contra]

    # Find contralateral FOs
    ipsi = ic[sides == "ipsilateral"]
    ipsi = ipsi[ipsi < mins[-1]]
    tc_contra = [int(mins[mins > x][0]) for x in ipsi]

    tc_ipsi = list(set(tc_ipsi))
    tc_contra = list(set(tc_contra))

    # combine
    toe_off = np.array(tc_ipsi + tc_contra).astype(int)
    toe_off_side = np.array(["ipsilateral" for x in tc_ipsi] + ["contralateral" for x in tc_contra])

    # sort
    toe_off_side = toe_off_side[toe_off.argsort()]
    toe_off = toe_off[toe_off.argsort()]
    return toe_off, toe_off_side


def apply_adapted_diao(data, window, side, selectFreq=True, checkSSA=False):
    # Load SI and ML signals
    acc_si = lp_filter(data[:, 1], 5)
    acc_ml = lp_filter(data[:, 0], 5)
    # Perform SSA on SI and ML signals (SI first)
    ssa_si_axis = SSA(acc_si, window, save_mem=False)
    ssa_ml_axis = SSA(acc_ml, window, save_mem=False)
    # Decide which reconstructed components to choose
    if selectFreq:
        # Select suitable reconstructed components
        acc_ssa_si, _ = check_rc_frequency(ssa_si_axis, 1.2, 2.8)
        acc_ssa_ml_dominant, acc_ssa_ml = check_rc_frequency(ssa_ml_axis, 0.6, 1.4)
    else:
        # Original Diao approach of using 2nd RC
        acc_ssa_si = ssa_si_axis.reconstruct(1)
        acc_ssa_ml_dominant = ssa_ml_axis.reconstruct(1)
        acc_ssa_ml = ssa_ml_axis.reconstruct([x for x in range(0, ssa_ml_axis.L) if x != 1])


    # # Visual check that SSA was correct
    if checkSSA:
        check_ssa_results(ssa_si_axis, ssa_ml_axis, acc_ssa_si, acc_ssa_ml, acc_ssa_ml_dominant)

    # find gait events
    # ic, ic_sides = detect_ic(acc_ssa_si, acc_ssa_ml, window)
    ic, ic_sides = detect_ic_adapted(acc_ssa_si, acc_ssa_ml_dominant, window)
    tc, tc_sides = detect_tc(acc_ssa_ml, ic, ic_sides)
    # Find sides
    LICs_l, RICs_l = order_gait_events_with_side(ic, ic_sides, side)
    LTCs_l, RTCs_l = order_gait_events_with_side(tc, tc_sides, side)

    mean_corrected_si = lp_filter(data[:, 1], 12) - np.mean(lp_filter(data[:, 1], 12))#data[:, 1] - np.mean(data[:, 1])

    # fig, axs = plot_diao_summary(mean_corrected_si, acc_ssa_si, acc_ssa_ml, acc_ssa_ml_dominant, LICs_l,
    #                   RICs_l, LTCs_l, RTCs_l)

    fig, axs = [], []

    # fig, axs = plot_diao_demo(mean_corrected_si, acc_ssa_si, acc_ssa_ml, acc_ssa_ml_dominant, LICs_l,
    #                   RICs_l, LTCs_l, RTCs_l)

    # fig = plot_demo_si(mean_corrected_si, acc_ssa_si, LICs_l, RICs_l, LTCs_l, RTCs_l)
    # fig.show()
    # fig = plot_demo_ml(acc_ssa_ml, acc_ssa_ml_dominant, LICs_l, RICs_l, LTCs_l, RTCs_l)
    # fig.show()
    # axs = []

    return LICs_l, RICs_l, LTCs_l, RTCs_l, fig, axs


def process_tug_trials(selectFreq=True):
    tugEventsDF = pd.read_csv("../utils/info/TUG Gait Events.csv", index_col="Filename")
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
            if (subjectNum > 0) and (subjectNum not in [46, 47, 48]):
                # if (subjectNum == 52 and trialNum == 24) and (subjectNum not in [46, 47, 48]):
                print(imuFilename)
                imuDF = pd.read_csv(os.path.join("../imu/data/TF_{:02d}".format(subjectNum), imuFilename))
                # linAccResolved = calculate_tilt_corrected_linear_acceleration_adapted(imuDF, gVal=9.81)
                imuDF = imuDF.loc[firstStepIdx + offset:tugEventsDF.at[
                    opticalFilename, "EndWalkIdx"] + offset + 5]  # Align IMU and optical systems
                for sideNum, side in enumerate(["lear"]):
                    data = imuDF[["AccY" + side, "AccZ" + side]].to_numpy()
                    LICs_l, RICs_l, LTCs_l, RTCs_l, fig, axs = apply_adapted_diao(data, 100, side, selectFreq, False)

                    # add these to df
                    if selectFreq:
                        eventsDFPath = "data/TUGDiaoEventsFFT.csv"
                    else:
                        eventsDFPath = "data/TUGDiaoEventsOriginal.csv"

                    diaoEventsDF = pd.read_csv(eventsDFPath, index_col="Filename", dtype="object")
                    diaoEventsDF.loc[opticalFilename, "Left ICs"] = LICs_l + firstStepIdx
                    diaoEventsDF.loc[opticalFilename, "Right ICs"] = RICs_l + firstStepIdx
                    diaoEventsDF.loc[opticalFilename, "Left FOs"] = LTCs_l + firstStepIdx
                    diaoEventsDF.loc[opticalFilename, "Right FOs"] = RTCs_l + firstStepIdx
                    diaoEventsDF.to_csv(eventsDFPath)
                    ##################

                    if turnsIdxs is not None:
                        axs[0, 0].vlines(turnsIdxs[:-1], axs[0, 0].get_ylim()[0], axs[0, 0].get_ylim()[1], color='y', linestyle='-.')
                        axs[0, 1].vlines(turnsIdxs[:-1], axs[0, 1].get_ylim()[0], axs[0, 1].get_ylim()[1], color='y', linestyle='-.')
                        axs[1, 0].vlines(turnsIdxs[:-1], axs[1, 0].get_ylim()[0], axs[1, 0].get_ylim()[1], color='y', linestyle='-.')
                        axs[1, 1].vlines(turnsIdxs[:-1], axs[1, 1].get_ylim()[0], axs[1, 1].get_ylim()[1], color='y', linestyle='-.')

                    # Adjust x tick values
                    for i in [0, 1]:
                        for j in [0, 1]:
                            currentLabels = axs[i, j].get_xticks()
                            axs[i, j].set_xticklabels([int(x) for x in currentLabels + int(firstStepIdx + int(offset))])
                    #         axs[i, j].set_xticklabels([x for x in range(int(firstStepIdx + offset), int(tugEventsDF.at[
                    # opticalFilename, "EndWalkIdx"] + offset + 6))])

                    fig.suptitle("{} - {}".format(opticalFilename.split(".")[0], side), fontsize=20)
                    fig.tight_layout()
                    fig.subplots_adjust(top=0.88)
                    plt.savefig("../plots/{}/{}.png".format("TUG/Diao/", opticalFilename.split(".")[0]))
                    # plt.show()
                    plt.close()


def process_nod_shake_trials(activity, selectFreq):
    offsetDF = pd.read_csv("../utils/info/IMU Offsets.csv", index_col="Filename")
    for subjectNum in [x for x in range(21, 22) if x not in [46, 47, 48]]:
        for trialNum in [4]:#get_generic_trial_nums(subjectNum, [activity]):
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
                        LICs_l, RICs_l, LTCs_l, RTCs_l, fig, axs = apply_adapted_diao(data, 200, side, selectFreq, False)
                    except:
                        LICs_l, RICs_l, LTCs_l, RTCs_l, fig, axs = apply_adapted_diao(data, 100, side, selectFreq,
                                                                                      False)
                    # add these to df
                    if selectFreq:
                        eventsDFPath = "data/{}DiaoEventsFFT.csv".format(activity)
                    else:
                        eventsDFPath = "data/{}DiaoEventsOriginal.csv".format(activity)
                    diaoEventsDF = pd.read_csv(eventsDFPath, index_col="Filename",
                                               dtype="object")

                    # diaoEventsDF.loc[opticalFilename, "Left ICs"] = LICs_l
                    # diaoEventsDF.loc[opticalFilename, "Right ICs"] = RICs_l
                    # diaoEventsDF.loc[opticalFilename, "Left FOs"] = LTCs_l
                    # diaoEventsDF.loc[opticalFilename, "Right FOs"] = RTCs_l

                    diaoEventsDF.to_csv(eventsDFPath)
                    # ##################



                    # fig.suptitle("{} - {}".format(imuFilename.split(".")[0], side), fontsize=20)
                    # fig.tight_layout()
                    # fig.subplots_adjust(top=0.88)
                    # plt.savefig()
                    plt.show()
                    # plt.savefig("../plots/{}/{}.png".format(activity, opticalFilename.split(".")[0]))
                    # plt.close()
            except Exception as e:
                print(e)


if __name__ == "__main__":
    # # Firstly process TUG trials
    # process_tug_trials(selectFreq=True)
    process_nod_shake_trials("Walk",  selectFreq=True)
    # create_events_df("WalkSlow", "WalkSlowDIaoEventsFFT.csv")

