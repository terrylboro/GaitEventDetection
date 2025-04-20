# A script to implement the singular spectrum analysis method described by Jarchi et al.
# https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6710137
# Handy Python tutorial here: https://www.kaggle.com/code/jdarcy/introducing-ssa-for-time-series-decomposition
# Class is simply lifted from this tutorial

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils.data_manipulation_utils import split_imu_df_by_sensor, calculate_acc_zero


class SSA(object):
    __supported_types = (pd.Series, np.ndarray, list)

    def __init__(self, tseries, L, save_mem=True):
        """
        Decomposes the given time series with a singular-spectrum analysis. Assumes the values of the time series are
        recorded at equal intervals.

        Parameters
        ----------
        tseries : The original time series, in the form of a Pandas Series, NumPy array or list.
        L : The window length. Must be an integer 2 <= L <= N/2, where N is the length of the time series.
        save_mem : Conserve memory by not retaining the elementary matrices. Recommended for long time series with
            thousands of values. Defaults to True.

        Note: Even if an NumPy array or list is used for the initial time series, all time series returned will be
        in the form of a Pandas Series or DataFrame object.
        """

        # Tedious type-checking for the initial time series
        if not isinstance(tseries, self.__supported_types):
            raise TypeError("Unsupported time series object. Try Pandas Series, NumPy array or list.")

        # Checks to save us from ourselves
        self.N = len(tseries)
        if not 2 <= L <= self.N / 2:
            raise ValueError("The window length must be in the interval [2, N/2].")

        self.L = L
        self.orig_TS = pd.Series(tseries)
        self.K = self.N - self.L + 1

        # Embed the time series in a trajectory matrix
        self.X = np.array([self.orig_TS.values[i:L + i] for i in range(0, self.K)]).T

        # Decompose the trajectory matrix
        self.U, self.Sigma, VT = np.linalg.svd(self.X)
        self.d = np.linalg.matrix_rank(self.X)

        self.TS_comps = np.zeros((self.N, self.d))

        if not save_mem:
            # Construct and save all the elementary matrices
            self.X_elem = np.array([self.Sigma[i] * np.outer(self.U[:, i], VT[i, :]) for i in range(self.d)])

            # Diagonally average the elementary matrices, store them as columns in array.
            for i in range(self.d):
                X_rev = self.X_elem[i, ::-1]
                self.TS_comps[:, i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0] + 1, X_rev.shape[1])]

            self.V = VT.T
        else:
            # Reconstruct the elementary matrices without storing them
            for i in range(self.d):
                X_elem = self.Sigma[i] * np.outer(self.U[:, i], VT[i, :])
                X_rev = X_elem[::-1]
                self.TS_comps[:, i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0] + 1, X_rev.shape[1])]

            self.X_elem = "Re-run with save_mem=False to retain the elementary matrices."

            # The V array may also be very large under these circumstances, so we won't keep it.
            self.V = "Re-run with save_mem=False to retain the V matrix."

        # Calculate the w-correlation matrix.
        self.calc_wcorr()

    def components_to_df(self, n=0):
        """
        Returns all the time series components in a single Pandas DataFrame object.
        """
        if n > 0:
            n = min(n, self.d)
        else:
            n = self.d

        # Create list of columns - call them F0, F1, F2, ...
        cols = ["F{}".format(i) for i in range(n)]
        return pd.DataFrame(self.TS_comps[:, :n], columns=cols, index=self.orig_TS.index)

    def reconstruct(self, indices):
        """
        Reconstructs the time series from its elementary components, using the given indices. Returns a Pandas Series
        object with the reconstructed time series.

        Parameters
        ----------
        indices: An integer, list of integers or slice(n,m) object, representing the elementary components to sum.
        """
        if isinstance(indices, int): indices = [indices]

        ts_vals = self.TS_comps[:, indices].sum(axis=1)
        return pd.Series(ts_vals, index=self.orig_TS.index)

    def calc_wcorr(self):
        """
        Calculates the w-correlation matrix for the time series.
        """

        # Calculate the weights
        w = np.array(list(np.arange(self.L) + 1) + [self.L] * (self.K - self.L - 1) + list(np.arange(self.L) + 1)[::-1])

        def w_inner(F_i, F_j):
            return w.dot(F_i * F_j)

        # Calculated weighted norms, ||F_i||_w, then invert.
        F_wnorms = np.array([w_inner(self.TS_comps[:, i], self.TS_comps[:, i]) for i in range(self.d)])
        F_wnorms = F_wnorms ** -0.5

        # Calculate Wcorr.
        self.Wcorr = np.identity(self.d)
        for i in range(self.d):
            for j in range(i + 1, self.d):
                self.Wcorr[i, j] = abs(w_inner(self.TS_comps[:, i], self.TS_comps[:, j]) * F_wnorms[i] * F_wnorms[j])
                self.Wcorr[j, i] = self.Wcorr[i, j]

    def plot_wcorr(self, min=None, max=None):
        """
        Plots the w-correlation matrix for the decomposed time series.
        """
        if min is None:
            min = 0
        if max is None:
            max = self.d

        if self.Wcorr is None:
            self.calc_wcorr()

        axWCorr = plt.imshow(self.Wcorr)
        plt.xlabel(r"$\tilde{F}_i$")
        plt.ylabel(r"$\tilde{F}_j$")
        plt.colorbar(axWCorr.colorbar, fraction=0.045)
        axWCorr.colorbar.set_label("$W_{i,j}$")
        plt.clim(0, 1)

        # For plotting purposes:
        if max == self.d:
            max_rnge = self.d - 1
        else:
            max_rnge = max

        plt.xlim(min - 0.5, max_rnge + 0.5)
        plt.ylim(max_rnge + 0.5, min - 0.5)
        plt.title("W-Correlation Matrix")

    def plot_eigenvalues(self):
        """
        Plot the eigenvalues (sigma) from SVD and the relative contributions.
        """
        sigma_sumsq = (self.Sigma ** 2).sum()
        figEig, axEig = plt.subplots(1, 2, figsize=(14, 5))
        axEig[0].stem(self.Sigma ** 2 / sigma_sumsq * 100)
        axEig[0].set_xlim(-0.5, 11.5)
        axEig[0].set_title("Relative Contribution of $\mathbf{X}_i$ to Trajectory Matrix")
        axEig[0].set_xlabel("$i$")
        axEig[0].set_ylabel("Contribution (%)")
        axEig[1].plot((self.Sigma ** 2).cumsum() / sigma_sumsq * 100)
        axEig[1].set_xlim(-0.5, 11.5)
        axEig[1].set_title("Cumulative Contribution of $\mathbf{X}_i$ to Trajectory Matrix")
        axEig[1].set_xlabel("$i$")
        axEig[1].set_ylabel("Contribution (%)")

    def plot_elemental_matrices(self, n):
        if not isinstance(self.X_elem, str):
            n = min(n, self.d)
            nRows, nCols = int(np.ceil(n/4)) , 4
            figEl, axEl = plt.subplots(nRows, nCols)
            for row in range(nRows):
                for col in range(nCols):
                    axEl[row, col].imshow(self.X_elem[row*4 + col])
                    axEl[row, col].set_title("$\mathbf{X}_{" + str(row*4 + col) + "}$")
            # plt.tight_layout()
        else:
            raise ValueError("Must re-run with save_mem=False!")

    def plot_individual_reconstructed_signals(self, indices):
        """
        Subplots which show the reconstructed components.
        :return:
        """
        numRows = len(indices)
        
        # Plot on 3 sets of axes
        figRC, axRC = plt.subplots(numRows, 1)
        for i in range(len(indices)):
            axRC[i].plot(self.TS_comps[:, indices[i]], color='r', label="RC"+str(indices[i]))
            axRC[i].legend()
        plt.suptitle("Reconstructed Components")

    def plot_cumulative_reconstructed_signals(self, level1Idx, level2Idx, title):
        """
        Subplots which show the total approximated signal when accumulating reconstructed components.
        :param level1Idx:
        :param level2Idx:
        :return:
        """
        total_reconstruction = self.reconstruct([x for x in range(self.L)])
        level1_reconstruction = self.reconstruct([x for x in range(0, level1Idx)])
        level2_reconstruction = self.reconstruct([x for x in range(0, level2Idx)])
        level3_reconstruction = self.reconstruct([x for x in range(level2Idx, self.L)])

        # Plot on 3 sets of axes
        figCRC, axCRC = plt.subplots(4, 1, figsize=(14, 5))
        for i, (reconstruction, RCs) in enumerate([(total_reconstruction, "Total Reconstruction"),
                                                   (level1_reconstruction, "RCs 0 to {}".format(level1Idx)),
                                                   (level2_reconstruction,
                                                    "RCs 0 to {}".format(level2Idx)),
                                                   (level3_reconstruction, "RCs {} to end".format(level2Idx))
                                                   ]):
            axCRC[i].plot(reconstruction, color='r', label=RCs)
            if i == 0:
                axCRC[i].plot(self.orig_TS, color='b', label='Original Signal', alpha=0.4)
            else:
                axCRC[i].plot(self.orig_TS, color='b', label='_Original', alpha=0.4)
            axCRC[i].legend()
        plt.suptitle(title+" Reconstructed Components vs Original TS")

    def plot_grouped_reconstruction(self, groupsList):
        """
        Plot grouped reconstructed components, the idea being to illustrate the different contributions to the signal.
        :param groupsList: List of lists containing indices to group together
        :return: None
        """
        # Plot on 3 sets of axes
        figGRC, axGRC = plt.subplots(len(groupsList), 1)
        for i in range(len(groupsList)):
            reconstruction = self.reconstruct(groupsList[i])
            # axGRC[i].plot(reconstruction, color='r', label="RCs "+",".join(str(x) for x in groupsList[i]))
            axGRC[i].plot(reconstruction, color='r', label="RCs {}-{}".format(groupsList[i][0], groupsList[i][-1]))
            axGRC[i].legend()
        plt.suptitle("Grouped Reconstructed Components")

    def plot_origTS(self):
        figOrigTS = plt.figure()
        self.orig_TS.plot()
        plt.title("Original Timeseries (Normalised)")
        plt.xlabel("Samples (100 samples = 1 second)")
        plt.ylabel("Acceleration / $ms^{-1}$")


#############################################################################
def ssa(data):
    L = 70  # window length / embedding dimension
    N = len(data)

    # Normalise the data and remove mean
    data = (data - np.mean(data)) / np.std(data)

    # Calculate the covariance matrix, X (trajectory)
    # time series s of length n is converted to an L x N matrix (trajectory matrix), X
    K = N - L + 1  # The number of columns in the trajectory matrix.
    X = np.column_stack([data[i:i+L] for i in range(0, K)])
    # Note: the i+L above gives us up to i+L-1, as numpy array upper bounds are exclusive.
    display_traj_mat(X)

    # Decomposition of trajectory matrix
    d = np.linalg.matrix_rank(X)  # The intrinsic dimensionality of the trajectory space.
    U, Sigma, V = np.linalg.svd(X)
    V = V.T  # Note: the SVD routine returns V^T, not V
    X_elem = np.array([Sigma[i] * np.outer(U[:, i], V[:, i]) for i in range(0, d)])
    # Quick sanity check: the sum of all elementary matrices in X_elm should be equal to X, to within a
    # *very small* tolerance:
    if not np.allclose(X, X_elem.sum(axis=0), atol=1e-10):
        print("WARNING: The sum of X's elementary matrices is not equal to X!")

    n = min(12, d)  # In case d is less than 12 for the toy series. Say, if we were to exclude the noise component...
    for i in range(n):
        plt.subplot(4, 4, i + 1)
        title = "$\mathbf{X}_{" + str(i) + "}$"
        plot_2d(X_elem[i], title)
    plt.tight_layout()

    sigma_sumsq = (Sigma ** 2).sum()
    import matplotlib as mpl
    mpl.use("Qt5Agg")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].stem(Sigma ** 2 / sigma_sumsq * 100, lw=2.5)
    ax[0].set_xlim(0, 11)
    ax[0].set_title("Relative Contribution of $\mathbf{X}_i$ to Trajectory Matrix")
    ax[0].set_xlabel("$i$")
    ax[0].set_ylabel("Contribution (%)")
    ax[1].stem((Sigma ** 2).cumsum() / sigma_sumsq * 100, lw=2.5)
    ax[1].set_xlim(0, 11)
    ax[1].set_title("Cumulative Contribution of $\mathbf{X}_i$ to Trajectory Matrix")
    ax[1].set_xlabel("$i$")
    ax[1].set_ylabel("Contribution (%)")

    plt.show()


def display_traj_mat(X):
    ax = plt.matshow(X)
    plt.xlabel("$L$-Lagged Vectors")
    plt.ylabel("$K$-Lagged Vectors")
    plt.colorbar(ax.colorbar, fraction=0.025)
    ax.colorbar.set_label("$F(t)$")
    plt.title("The Trajectory Matrix for the Toy Time Series")
    # plt.show()


# A simple little 2D matrix plotter, excluding x and y labels.
def plot_2d(m, title=""):
    plt.imshow(m)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)


def main():
    from utils.data_manipulation_utils import lp_filter
    from utils.orientation_utils import calculate_corrected_imu_orientation_timeseries, \
        calculate_tilt_corrected_linear_acceleration
    from utils.participant_info_utils import get_generic_trial_nums
    import os
    activity = "Sit2Stand"
    # load the imu data
    for subjectNum in [x for x in range(56, 68) if x not in [100]]:
        # direction to imu data
        trialNums = get_generic_trial_nums(subjectNum, [activity])
        # get the corresponding files
        imuFilenames = []
        for trialNum in trialNums:
            imuFilenames.append("TF_{}-{}.csv".format(str(subjectNum).zfill(2), str(trialNum).zfill(2)))

        for trialNum, imu in enumerate(imuFilenames):
            print(imu + " :  " + str(trialNum))
            # Load and sort IMU data by location
            imuDF = pd.read_csv(os.path.join("../imu/data/TF_{}".format(str(subjectNum).zfill(2)), imu))
            dfL, dfR, dfC, dfP = split_imu_df_by_sensor(imuDF)

            # Calculate the orientations for each location
            lCorrectedOrientationEuler = calculate_corrected_imu_orientation_timeseries(dfL.to_numpy(), subjectNum,
                                                                                        "lear")
            rCorrectedOrientationEuler = calculate_corrected_imu_orientation_timeseries(dfR.to_numpy(), subjectNum,
                                                                                        "rear")
            cCorrectedOrientationEuler = calculate_corrected_imu_orientation_timeseries(dfC.to_numpy(), subjectNum,
                                                                                        "chest")
            # pCorrectedOrientationEuler = calculate_corrected_imu_orientation_timeseries(dfP.to_numpy(), subjectNum,
            #                                                                             "pocket")

            # Now perform the analysis here !
            pitchArr = np.concatenate((np.expand_dims(lCorrectedOrientationEuler[:, 1], axis=1),
                                       np.expand_dims(rCorrectedOrientationEuler[:, 1], axis=1),
                                       np.expand_dims(cCorrectedOrientationEuler[:, 1], axis=1)), axis=1)
            # Other papers work with sin(theta)
            sinPitchArr = np.sin(lp_filter(np.sin(pitchArr * np.pi / 180), 15))

            # Identify the peak flexion point, then the start/end are the troughs either side
            tPt, valPt = np.argmin(sinPitchArr, axis=0), np.amin(sinPitchArr, axis=0)
            tP1, valP1 = np.argmax(sinPitchArr[:int(tPt[2]), :], axis=0), np.amax(sinPitchArr[:int(tPt[2]), :],
                                                                                  axis=0)
            tP2, valP2 = np.argmax(sinPitchArr[int(tPt[2]):, :], axis=0) + tPt[2], np.amax(
                sinPitchArr[int(tPt[2]):, :], axis=0)

            # Having segmented the trial, now analyse the actual movement data
            learAcc, _, learResolvedAcc, learGVec = calculate_tilt_corrected_linear_acceleration(imuDF, "lear",
                                                                                                 includeTimestamp=False,
                                                                                                 plot=False)
            rearAcc, _, rearResolvedAcc, rearGVec = calculate_tilt_corrected_linear_acceleration(imuDF, "rear",
                                                                                                 includeTimestamp=False,
                                                                                                 plot=False)
            chestAcc, _, chestResolvedAcc, chestGVec = calculate_tilt_corrected_linear_acceleration(imuDF, "chest",
                                                                                                    includeTimestamp=False,
                                                                                                    plot=False)
            # Crop this down to start/stop indices found earlier
            learAcc, learGVec, learResolvedAcc = learAcc[tP1[0]:tP2[0], :], learGVec[tP1[0]:tP2[0],
                                                                            :], learResolvedAcc[tP1[0]:tP2[0], :]
            learAccArr = np.concatenate((learAcc, learGVec, learResolvedAcc), axis=1)
            rearAcc, rearGVec, rearResolvedAcc = rearAcc[tP1[1]:tP2[1], :], rearGVec[tP1[1]:tP2[1],
                                                                            :], rearResolvedAcc[tP1[1]:tP2[1], :]
            rearAccArr = np.concatenate((rearAcc, rearGVec, rearResolvedAcc), axis=1)
            chestAcc, chestGVec, chestResolvedAcc = chestAcc[tP1[2]:tP2[2], :], chestGVec[tP1[2]:tP2[2],
                                                                                :], chestResolvedAcc[tP1[2]:tP2[2],
                                                                                    :]
            chestAccArr = np.concatenate((chestAcc, chestGVec, chestResolvedAcc), axis=1)

            window = 70
            title_order = ['AP Axis', 'SI Axis']
            # demo_ssa = SSA(data, window)
            # demo_ssa.plot_wcorr()
            # plt.title("W-Correlation for Walking Time Series")

            for i, (sensorType, sensorAxis, sensorDF, side) in enumerate([("Acc", "Z", dfC, "chest"), ("Acc", "X", dfL, "left")]):
                # ssa(imuDF["AccZlear"].to_numpy())
                # begin reconstructing the components etc.
                # plt.figure(i)
                # perform SSA on given axis
                # sensorDF = dfC
                # sensorType = "Gyro"

                # plt.figure()
                # dfC["Acc" + side + sensorLocation].plot()
                # plt.title("Original data")
                # plt.show()

                try:

                    data = sensorDF["GyroY"].to_numpy()#calculate_acc_zero(sensorDF[["GyroX", "GyroY", "GyroZ"]].to_numpy())
                    data = (data - np.mean(data)) / np.std(data)
                    axis_data = SSA(data, window, save_mem=False)


                    # normalisedTSdata = (sensorDF[sensorType+sensorAxis] - sensorDF[sensorType+sensorAxis].mean()) / sensorDF[sensorType+sensorAxis].std()
                    # axis_data = SSA(normalisedTSdata, window, save_mem=False)

                    # import matplotlib as mpl
                    # mpl.use("Qt5Agg")
                    # axis_data.plot_wcorr()
                    # axis_data.plot_eigenvalues()
                    # axis_data.plot_elemental_matrices(12)
                    # axis_data.plot_individual_reconstructed_signals(range(0, 5))
                    axis_data.plot_cumulative_reconstructed_signals(2, 5, imu+" "+side)
                    # axis_data.plot_origTS()
                    # axis_data.plot_grouped_reconstruction([[0, 1, 2], [3, 4, 5], [x for x in range(6, 13)],
                    #                                        [13, 14, 15], [x for x in range(16, 25)], [x for x in range(29, 43)]])
                    plt.savefig("../sit2stand/SSAGyroFigures/"+imu.split(".")[0]+"-"+side+".png")
                    plt.close()
                except ValueError as e:
                    print(e)


if __name__ == '__main__':
    main()