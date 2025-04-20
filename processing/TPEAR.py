import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# from utils.s2s_utils import *
from utils.data_manipulation_utils import lp_filter, bp_filter
from utils.orientation_utils import calculate_tilt_corrected_linear_acceleration_adapted
from utils.gait_utils import *
from processing.ssa import SSA

import matplotlib as mpl
mpl.use('Qt5Agg')


def sharpen_peaks(data, weight=0.2):
    """
    Apply peak sharpening to inputted data. Adjust the level of sharping using weight (0 to 1).
    :param data: Smoothed acceleration signal.
    :param weight: Sharpening level coefficient.
    :return: Sharpened signal.
    """
    # Find signal double derivative
    dataDoubleDiff = np.diff(np.diff(data.flatten()))
    return np.insert(data[2:] - weight * dataDoubleDiff, 2, data[0:2])


def find_ic_side_from_dominant_ml(acc_ml_dominant, peaks_ic):
    """
    Assume the IC side matches that of the dominant ML signal (Seifer 2023).
    :param acc_ml_dominant:
    :param peaks_ic:
    :return: Array containing predicted sides for each IC.
    """
    import numpy as np
    # Find corresponding sides
    sides = []
    for peakDiff in (acc_ml_dominant.diff().loc[peaks_ic + 1]):
        if peakDiff > 0:
            sides.append("contralateral")
        else:
            sides.append("ipsilateral")
    return np.array(sides)


def handle_more_than_2_peaks(dominantPeak, si_raw, windowPeaks, windowStart):
    """
    Handle the case where more than 2 peaks appear in the window.
    This is used on multiple occasions in the algorithm hence is implemented as a reusable function.
    :param dominantPeak: Location of dominant SI peak.
    :param si_raw: SI acceleration signal.
    :param windowPeaks: Indices of the peaks in the window under consideration.
    :param windowStart: Index where the window starts.
    :return: Predicted IC and TC event.
    """
    import numpy as np
    # select IC as closest peak to dominant one, unless this is the last peak (extremely fringe case)
    peakNearestDominant = np.argmin(np.abs(windowPeaks - (dominantPeak - windowStart)))
    if peakNearestDominant != len(windowPeaks) - 1:
        IC = windowPeaks[peakNearestDominant] + windowStart
        # Cycle back from last peak to find FO
        foList = windowPeaks[peakNearestDominant + 1:]
        # Handle strange aberration where suddenly there are no more peaks
        if len(foList) == 0:
            FO = IC
            while si_raw[FO] > 0:
                FO += 1
        else:
            FO = None
            for fo in foList[::-1]:
                if si_raw[fo + windowStart] > 0:
                    FO = fo + windowStart
                    break
            # Worst case, just pick the zero crossing
            if FO is None:
                FO = IC
                while si_raw[FO] > 0:
                    FO += 1
    else:
        # In the hypothetical fringe case this is basically just a 2-peak situation.
        IC = windowPeaks[-2] + windowStart
        FO = windowPeaks[-1] + windowStart
    return IC, FO


def perform_si_windowing(acc_si_dominant, si_raw, sample_rate_hz: float):
    from scipy.signal import find_peaks
    import numpy as np
    import matplotlib.pyplot as plt  # include this for plotting
    # PREPROCESSING: find minimum on SI axis with removed trend. Peaks should correspond to IC
    peaks_ic, _ = find_peaks(acc_si_dominant, distance=sample_rate_hz * 0.2)
    IC_list = []
    FO_list = []

    for i in range(len(peaks_ic)):
        # OPEN WINDOW: Only interested in window just before predicted IC until negative dominant IC
        # At the end of the window, the participant is almost definitely in swing phase (hence TC has happened)
        windowStart = peaks_ic[i]
        while acc_si_dominant[windowStart] > 0.8 * acc_si_dominant[peaks_ic[i]] and windowStart > 0:
            windowStart -= 1
        if i == len(peaks_ic)-1:
            windowEnd = np.argmin(acc_si_dominant[peaks_ic[i]:]) + peaks_ic[i]
        else:
            windowEnd = np.argmin(acc_si_dominant[peaks_ic[i]:peaks_ic[i+1]]) + peaks_ic[i]
        windowPeaks, _ = find_peaks(si_raw[windowStart:windowEnd])

        # Handle cases by # peaks detected. Expected # is 2 according to theory but this varies
        if len(windowPeaks) == 2:
            # Take the last 2 peaks as IC and FO
            FO = windowPeaks[-1] + windowStart
            IC = windowPeaks[-2] + windowStart
        elif len(windowPeaks) > 2:
            IC, FO = handle_more_than_2_peaks(peaks_ic[i], si_raw, windowPeaks, windowStart)
        elif len(windowPeaks) == 1:
            # This can happen when the peaks merge or the participants takes a step on their toes.
            # This is more common during dynamic turns with young participants moving quickly.
            if len(si_raw) - (windowPeaks[-1] + windowStart) < 15:
                # This handles the case where the IC occurs very near the end
                IC = windowPeaks[-1] + windowStart
                FO = len(si_raw) - 1  # this will be excluded from comparison anyway
            else:
                # APPLY SHARPENING then recheck the peaks
                sharpPeaks, _ = find_peaks(sharpen_peaks(si_raw[windowStart:windowEnd], weight=0.6), distance=8)

                if len(sharpPeaks) > 2:
                    IC, FO = handle_more_than_2_peaks(peaks_ic[i], si_raw, sharpPeaks, windowStart)
                elif len(sharpPeaks) == 2:
                    FO = sharpPeaks[-1] + windowStart
                    IC = sharpPeaks[-2] + windowStart
                else:
                    # In this case, the usual two peaks have likely merged
                    croppedSignal = si_raw[windowStart:windowEnd]
                    if len(croppedSignal[croppedSignal > 0]) > 1:
                        # Remove the negative parts of the signal provided this remains nonzero
                        croppedSignal = croppedSignal[croppedSignal > 0]
                    grad = np.diff(sharpen_peaks(croppedSignal, weight=0.6))

                    # We want to pick FO as shallow point on descending gradient
                    # But this must be after the IC and within the remaining window
                    IC = windowPeaks[-1] + windowStart
                    if windowPeaks[-1]+2 < len(grad)-1:
                        grad = grad[windowPeaks[-1]+2:]  # add +2 to avoid double detection
                    elif windowPeaks[-1] > len(grad)-1:
                        grad = grad[-1]
                    else:
                        grad = grad[windowPeaks[-1]:]

                    FO = np.argmin(np.abs(grad)) + 1 + windowStart + windowPeaks[-1]

        else:
            # Shouldn't reach this state anyway
            IC = peaks_ic[i]
            FO = windowEnd

        IC_list.append(int(IC))
        FO_list.append(int(FO))

    return np.array(IC_list), np.array(FO_list)


def apply_tp_ear(data, window, selectFreq=True, checkSSA=False):
    # Find initial peak
    # Load SI and ML signals
    acc_si = bp_filter(data[:, 1], 0.5, 12)
    acc_ml = lp_filter(data[:, 0], 20)

    # Perform SSA on SI and ML signals (SI first)
    ssa_si_axis = SSA(acc_si, window, save_mem=False)
    ssa_ml_axis = SSA(acc_ml, window, save_mem=False)
    # Decide which reconstructed components to choose
    if selectFreq:
        # Select suitable reconstructed components
        acc_ssa_si, _ = check_rc_frequency(ssa_si_axis, 1.2, 2.8)
        acc_ssa_ml_dominant, acc_ssa_ml = check_rc_frequency(ssa_ml_axis, 0.6, 1.4)
    else:
        # Original Diao approach of just using 2nd RC
        acc_ssa_si = ssa_si_axis.reconstruct(1)
        acc_ssa_ml_dominant = ssa_ml_axis.reconstruct(1)
        acc_ssa_ml = ssa_ml_axis.reconstruct([x for x in range(0, ssa_ml_axis.L) if x != 1])

    # Visual check that SSA was correct
    if checkSSA:
        check_ssa_results(ssa_si_axis, ssa_ml_axis, acc_ssa_si, acc_ssa_ml, acc_ssa_ml_dominant)

    # Add windowing to SI signal to select correct peak
    ic, tc = perform_si_windowing(acc_ssa_si, acc_si, 100)
    # print("ic: ", ic)
    # print("tc: ", tc)

    ic_sides = find_ic_side_from_dominant_ml(acc_ssa_ml_dominant, ic)
    # print(ic_sides)

    # find gait events
    tc_sides = []
    for side in ic_sides:
        if side == "contralateral":
            tc_sides.append("ipsilateral")
        else:
            tc_sides.append("contralateral")
    # print(tc_sides)

    # Find sides
    LICs_l, RICs_l = order_gait_events_with_side(ic, ic_sides, "lear")
    LTCs_l, RTCs_l = order_gait_events_with_side(tc, tc_sides, "lear")

    # # Uncomment the following to create plots
    # mean_corrected_si = data[:, 1] - np.mean(data[:, 1])
    # fig, axs = plot_diao_summary(mean_corrected_si, acc_ssa_si, acc_ssa_ml, acc_ssa_ml_dominant, LICs_l,
    #                              RICs_l, LTCs_l, RTCs_l)
    # fig, axs = plot_diao_demo(mean_corrected_si, acc_ssa_si, acc_ssa_ml, acc_ssa_ml_dominant, LICs_l,
    #                           RICs_l, LTCs_l, RTCs_l)
    # plt.show()

    fig, axs = [], []

    return LICs_l, RICs_l, LTCs_l, RTCs_l, fig, axs

