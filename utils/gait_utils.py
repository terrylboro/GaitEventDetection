def plot_diao_summary(si_data, acc_ssa_si, acc_ssa_ml, acc_ssa_ml_dominant, LICs_l, RICs_l, LTCs_l, RTCs_l):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(16, 6))
    
    # axs[0] = plt.subplot(221)
    axs[0, 0].plot(si_data, label='Original SI')
    axs[0, 0].plot(acc_ssa_si, label="Dominant SI")
    axs[0, 0].vlines(LICs_l, min(si_data), max(si_data), color='r', linestyle='solid',
               label="LICs")
    axs[0, 0].vlines(RICs_l, min(si_data), max(si_data), color='g', linestyle='solid',
               label="RICs")
    axs[0, 0].legend(loc='best', fontsize=14)

    # ax2 = plt.subplot(222)
    axs[0, 1].plot(si_data, label='Original SI')
    axs[0, 1].plot(acc_ssa_si, label="Dominant SI")
    axs[0, 1].vlines(LTCs_l, min(si_data), max(si_data), color='r', linestyle='--',
               label="LFOs")
    axs[0, 1].vlines(RTCs_l, min(si_data), max(si_data), color='g', linestyle='--',
               label="RFOs")
    axs[0, 1].legend(loc='best', fontsize=14)

    # ax3 = plt.subplot(223)
    axs[1, 0].plot(acc_ssa_ml, label='ML trend removed')
    axs[1, 0].plot(acc_ssa_ml_dominant, label="Dominant ML")
    axs[1, 0].vlines(LICs_l, min(acc_ssa_ml), max(acc_ssa_ml), color='r', linestyle='solid', label="LICs")
    axs[1, 0].vlines(RICs_l, min(acc_ssa_ml), max(acc_ssa_ml), color='g', linestyle='solid', label="RICs")
    axs[1, 0].legend(loc='best', fontsize=14)

    # ax4 = plt.subplot(224)
    axs[1, 1].plot(acc_ssa_ml, label='ML trend removed')
    axs[1, 1].plot(acc_ssa_ml_dominant, label="Dominant ML")
    axs[1, 1].vlines(LTCs_l, min(acc_ssa_ml), max(acc_ssa_ml), color='r', linestyle='--', label="LFOs")
    axs[1, 1].vlines(RTCs_l, min(acc_ssa_ml), max(acc_ssa_ml), color='g', linestyle='--', label="RFOs")
    axs[1, 1].legend(loc='best', fontsize=14)

    return fig, axs


def plot_diao_demo(si_data, acc_ssa_si, acc_ssa_ml, acc_ssa_ml_dominant, LICs_l, RICs_l, LTCs_l, RTCs_l):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axs = plt.subplots(2, figsize=(16, 6), sharex=True)

    # Plot SI signal
    sns.lineplot(si_data, label='Original SI', ax=axs[0])
    sns.lineplot(acc_ssa_si, label="Dominant SI", ax=axs[0])
    sns.scatterplot(x=LICs_l, y=acc_ssa_si[LICs_l], color="r", marker="o", s=40, label="LICs", ax=axs[0], zorder=7)
    sns.scatterplot(x=RICs_l, y=acc_ssa_si[RICs_l], color="g", marker="o", s=40, label="RICs", ax=axs[0], zorder=7)
    axs[0].set_ylabel("SI Acceleration / $ms^{-2}$", fontsize=18)
    axs[0].legend(loc='best', fontsize=12)

    # Plot ML signal
    sns.lineplot(acc_ssa_ml, label='ML (trend removed)', ax=axs[1])
    sns.lineplot(acc_ssa_ml_dominant, label="Dominant ML", ax=axs[1])
    sns.scatterplot(x=LTCs_l, y=acc_ssa_ml[LTCs_l], color="r", marker="^", s=80, label="LFOs", ax=axs[1], zorder=7)
    sns.scatterplot(x=RTCs_l, y=acc_ssa_ml[RTCs_l], color="g", marker="^", s=80, label="RFOs", ax=axs[1], zorder=7)
    axs[1].set_ylabel("ML Acceleration / $ms^{-2}$", fontsize=18)
    axs[1].set_xlabel("Time / s", fontsize=18)
    axs[1].legend(loc='lower right', fontsize=12)

    # Add axis labels
    import re
    xTickVals = [int(re.sub(u"\u2212", "-", i.get_text())) for i in axs[1].get_xticklabels()[1:]]
    axs[1].set_xticks(xTickVals, [int(x / 100) for x in xTickVals], fontsize=12)
    fig.suptitle("Diao Algorithm with Seifer Improvement", fontsize=30)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.align_ylabels(axs)
    return fig, axs


def plot_demo_si(si_data, acc_ssa_si, LICs_l, RICs_l, LTCs_l, RTCs_l):
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Show the results for the first time series and its subseries
    fig = plt.figure(figsize=(16, 6))

    # Plot SI signal
    sns.lineplot(si_data, label='Original SI', linewidth=3)
    sns.lineplot(acc_ssa_si, label="Dominant SI", linewidth=3, linestyle='-.')
    # Diao ICs
    sns.scatterplot(x=LICs_l, y=acc_ssa_si[LICs_l], color="r", edgecolor=None, marker="o", s=180, label="Left ICs (Diao)", zorder=7)
    sns.scatterplot(x=RICs_l, y=acc_ssa_si[RICs_l], color="g", edgecolor=None, marker="o", s=180, label="Right ICs (Diao)", zorder=7)
    # New ICs for TF_21-04
    sns.scatterplot(x=[222], y=si_data[222], color="r", edgecolor=None, marker="*", s=360, label="Left ICs (TP-EAR)", zorder=7)
    sns.scatterplot(x=[166, 276], y=si_data[[166, 276]], edgecolor=None, color="g", marker="*", s=360, label="Right ICs (TP-EAR)", zorder=7)
    # New TCs for TF_21-04
    sns.scatterplot(x=[179, 290], y=si_data[[179, 290]], edgecolor=None, color="r", marker="^", s=180, label="Left TCs (TP-EAR)", zorder=7)
    sns.scatterplot(x=[231], y=si_data[231], color="g", edgecolor=None, marker="^", s=180, label="Right TCs (TP-EAR)", zorder=7)
    plt.xlabel("Time / s", fontsize=40)
    plt.ylabel("SI Acceleration / $ms^{-2}$", fontsize=40)
    plt.legend(loc='best', fontsize=18, ncol=2)

    # Shade in windows
    import numpy as np
    plt.fill_between([x for x in range(162, 197)], np.min(si_data)-0.5, np.max(si_data)+0.5, facecolor='gray', alpha=.3)
    plt.fill_between([x for x in range(218, 250)], np.min(si_data)-0.5, np.max(si_data)+0.5, facecolor='gray', alpha=.3)
    plt.fill_between([x for x in range(272, 306)], np.min(si_data)-0.5, np.max(si_data)+0.5, facecolor='gray', alpha=.3)

    xTickLocs, xTickVals = plt.xticks()
    xTickVals = [str(int(x / 100)) for x in xTickLocs]
    print(plt.xticks())
    plt.xticks(xTickLocs, xTickVals, fontsize=20)

    plt.xlim(150, 320)
    plt.ylim(-3, 6.9)
    # plt.xticks(xTickLocs, [str(int(x / 100)) for x in xTickVals], fontsize=12)

    # plt.title("SI Signal Gait Event Detection", fontsize=30)
    plt.tight_layout()
    return fig


def plot_demo_ml(acc_ssa_ml, acc_ssa_ml_dominant, LICs_l, RICs_l, LTCs_l, RTCs_l):
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Show the results for the first time series and its subseries
    fig = plt.figure(figsize=(16, 6))

    # Plot ML signal
    sns.lineplot(acc_ssa_ml, label='ML (Trend Removed)', linewidth=3)
    sns.lineplot(acc_ssa_ml_dominant, label="Dominant ML", linewidth=3, linestyle='-.')

    plt.vlines(LICs_l, -2, 3, color="r", linestyles='dotted', linewidth=3, label='Left ICs')
    plt.vlines(RICs_l, -2, 3, color="g", linestyles='dotted', linewidth=3, label='Right ICs')

    offset=3
    QV2 = plt.quiver(222 - offset, acc_ssa_ml_dominant[222 - offset], 3,
                     (acc_ssa_ml_dominant[225 - offset] - acc_ssa_ml_dominant[222 - offset]),
                     angles='xy', scale_units='xy', scale=0.3,
                     zorder=7, color="r", label="Left Laterality")
    QV1 = plt.quiver(277-offset, acc_ssa_ml_dominant[277-offset], 3, (acc_ssa_ml_dominant[280-offset]-acc_ssa_ml_dominant[277-offset]), angles='xy', scale_units='xy', scale=0.3,
               zorder=7, color="g",  label="Right Laterality")
    QV3 = plt.quiver(168-offset, acc_ssa_ml_dominant[168-offset], 3, (acc_ssa_ml_dominant[171-offset] - acc_ssa_ml_dominant[168-offset]),
                     angles='xy', scale_units='xy', scale=0.3,
                     zorder=7, color="g", label="__no_legend__")

    sns.scatterplot(x=LTCs_l, y=acc_ssa_ml[LTCs_l], edgecolor=None, color="r", marker="^", s=180, label="Left TCs (Diao)", zorder=7)
    sns.scatterplot(x=RTCs_l, y=acc_ssa_ml[RTCs_l], edgecolor=None, color="g", marker="^", s=180, label="Right TCs (Diao)", zorder=7)
    plt.ylabel("ML Acceleration / $ms^{-2}$", fontsize=40)
    plt.xlabel("Time / s", fontsize=40)
    plt.legend(loc='lower right', fontsize=18, ncols=2)

    # Add axis labels
    xTickLocs, xTickVals = plt.xticks()
    xTickVals = [str(int(x / 100)) for x in xTickLocs]
    print(plt.xticks())
    plt.xticks(xTickLocs, xTickVals, fontsize=20)
    # Fix limits
    plt.xlim(150, 320)
    plt.ylim(-1.55, 1.3)
    # plt.title("ML Signal Gait Event Detection", fontsize=30)
    plt.tight_layout()
    return fig


def order_gait_events_with_side(eventArr, sideArr, sensorLocation):
    # print("Function arguments:\n{} - {} - {}".format(eventArr, sideArr, sensorLocation))
    leftList, rightList = [], []
    for i in zip(eventArr, sideArr):
        if i[1] == "contralateral":
            if sensorLocation == "lear" or sensorLocation == "chest":
                rightList.append(i[0])
            else:
                leftList.append(i[0])
        else:
            if sensorLocation == "lear" or sensorLocation == "chest":
                leftList.append(i[0])
            else:
                rightList.append(i[0])
    return leftList, rightList


def check_ssa_results(ssa_si_axis, ssa_ml_axis, acc_ssa_si, acc_ssa_ml, acc_ssa_ml_dominant, turnsIdxs=None):
    import matplotlib as mpl
    mpl.use("Qt5Agg")
    import matplotlib.pyplot as plt
    # Visualise SSA
    ssa_si_axis.plot_individual_reconstructed_signals(range(0, 5))
    plt.suptitle("SSA on SI")
    ssa_ml_axis.plot_individual_reconstructed_signals(range(0, 5))
    plt.suptitle("SSA on ML")
    # Plot outputted signals we have chosen
    fig1 = plt.figure()
    fig1.canvas.manager.window.move(200, 0)
    plt.plot(acc_ssa_si, label="SI dominant")
    plt.plot(acc_ssa_ml, label="ML")
    plt.plot(acc_ssa_ml_dominant, label="ML dominant")
    if turnsIdxs is not None:
        plt.vlines(turnsIdxs, min(acc_ssa_ml), max(acc_ssa_ml), color='y', linestyle='-.')
    plt.legend()
    plt.title("Reconstructed signals")
    plt.show()


def check_rc_frequency(ssa_axis, threshLow, threshHigh):
    from scipy.signal import find_peaks
    from scipy.fft import fft, fftfreq
    import numpy as np
    from matplotlib import pyplot as plt
    # Find correct component to reconstruct
    rcGoodFlag = False
    dominantRC = 1
    trendRCs = []

    for i in range(6):
        acc_ssa_dominant = ssa_axis.reconstruct(i)

        # Extra filtering method
        yf = fft(acc_ssa_dominant.to_numpy())
        N = len(acc_ssa_dominant)
        xf = fftfreq(N, (1 / 100))[:N // 2]
        freqPeak = xf[np.argmax(2.0 / N * np.abs(yf[0:N // 2]))]

        if freqPeak < threshLow:
            trendRCs.append(i)
        elif (threshHigh > freqPeak > threshLow) & (not rcGoodFlag):
            print("RC selected with freq: ", freqPeak)
            dominantRC = i
            rcGoodFlag = True

    acc_ssa_dominant = ssa_axis.reconstruct(dominantRC)
    acc_ssa = ssa_axis.reconstruct([x for x in range(0, ssa_axis.L) if x not in trendRCs])

    return acc_ssa_dominant, acc_ssa

