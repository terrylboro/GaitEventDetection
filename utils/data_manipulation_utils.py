def hp_filter(data, freq: float):
    """ Apply 2nd order high pass filter on inputted data with specified cut-off frequency.
    Filtering is performed in-place using scipy.filtfilt. """
    from scipy.signal import butter, filtfilt
    b, a = butter(2, freq, btype="high", fs=100, output='ba')
    return filtfilt(b, a, data, axis=0)


def lp_filter(data, freq: float):
    """ Apply 2nd order low pass filter on inputted data with specified cut-off frequency.
        Filtering is performed in-place using scipy.filtfilt. """
    from scipy.signal import butter, filtfilt
    b, a = butter(2, freq, btype="low", fs=100, output='ba')
    return filtfilt(b, a, data, axis=0)


def bp_filter(data, freq1: float, freq2: float):
    """ Apply 2nd order band pass filter on inputted data with specified cut-off frequencies.
        Filtering is performed in-place using scipy.filtfilt. """
    from scipy.signal import butter, filtfilt
    b, a = butter(2, [freq1, freq2], btype="band", fs=100, output='ba')
    return filtfilt(b, a, data, axis=0)


def normalize(v):
    """ Return normalised vector. """
    import numpy as np
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm


def normalise_0_to_1(arr):
    """
    Normalise each column of a 2D numpy array to values between 0 and 1, scaled by column min and max.
    :param arr: n x m numpy array
    :return: n x m normalised numpy array
    """
    import numpy as np
    normalisedArr = np.zeros_like(arr)
    if arr.ndim == 1:
        normalisedArr = (arr - arr.min()) / (arr.max() - arr.min())
    else:
        for i in range(arr.shape[-1]):
            normalisedArr[:, i] = (arr[:, i] - arr[:, i].min()) / (
                        arr[:, i].max() - arr[:,i].min())
    return normalisedArr

def unit_vector(vector):
    """ Returns the unit vector of the inputted vector. """
    import numpy as np
    return vector / np.linalg.norm(vector)


def find_nearest(array, value):
    """ Return index and value of inputted array which is nearest to the inputted value. """
    import numpy as np
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'. Example results:
    angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    angle_between((1, 0, 0), (1, 0, 0))
    0.0
    angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    :param v1: First vector
    :param v2: Second vector
    """
    import numpy as np
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def calculate_joint_angle(data):
    """
    Calculates absolute angle between 3 markers. Does not allow separation of X, Y, and Z.
    :param data: nx9 numpy array containing data from 3 markers
    :return: absolute angle defined from these markers in degrees
    """
    import numpy as np
    # 0 = TIB, 1 = ANK, 2 = TOE
    v1 = data[:, 0:3] - data[:, 3:6]
    v2 = data[:, 6:9] - data[:, 3:6]
    angle_arr = np.zeros((len(data), 1))
    for i in range(0, len(data)):
        angle_arr[i, 0] = angle_between(v1[i, :], v2[i, :])
    # angle_arr = angle_arr[~np.isnan(angle_arr)]
    angle_arr = angle_arr.reshape(len(angle_arr), 1)
    return np.rad2deg(angle_arr)


def calculate_acc_zero(data):
    """
    Calculate the resultant vector for given X, Y and Z data.
    :param data: nx3 numpy array
    :return: nx1 numpy array containing resultant of data
    """
    import numpy as np
    acc_zero_data = np.zeros((len(data), 1))
    for row in range(0, len(data)):
        acc_zero_data[row] = np.sqrt(np.sum(np.square(data[row, :])))
    return acc_zero_data.squeeze()


def project_vertical_component(data):
    """
    Find the vertical component (0, 0, 1) of a 3D vector in the global frame.
    :param data: n x 3 numpy array containing accelerometer data.
    :return: n x 1 numpy array containing vertical component of data vector.
    """
    import numpy as np
    data_s = calculate_acc_zero(data)
    data_u = np.zeros_like(data)
    angleArr = np.zeros((len(data), 1))
    vProjArr = np.zeros((len(data), 1))
    for i in range(0, len(data_u)):
        data_u[i, :] = unit_vector(data[i, :])
        angleArr[i] = np.clip(np.dot(data_u[i, :], np.array([0, 0, 1])), -1.0, 1.0)
        vProjArr[i] = data_s[i] * angleArr[i]
    return vProjArr, data_s, angleArr


def set_start_time_to_zero(df):
    """
    Index timing from start of trial. Works for multiple cols or single col.
    :param df: Pandas DataFrame containing trial data (including timestamp).
    :return: Pandas DataFrame containing trial data with timestamp indexed from zero.
    """
    timeCols = df.filter(like="Time").columns.tolist()
    df[timeCols] = df[timeCols] - df.loc[0, timeCols].values.squeeze()
    return df


def normalise_pd_series(df, colList=[]):
    """
    Normalise a Pandas DataFrame using min and max, returning in a range of 0,1.
    :param df: Pandas DataFrame containing data to be normalised.
    :param colList: Columns to be normalised.
    :return: Normalised Pandas Series or DataFrame.
    """
    return ((df[colList].abs() - df[colList].abs().min(axis=0)) /
            (df[colList].abs().max(axis=0) - df[colList].abs().min(axis=0)))


def first_nonzero(arr, axis, invalid_val=-1):
    import numpy as np
    mask = arr!=0
    # return first index for multidimensional array otherwise just an int
    if len(arr.shape) < 2:
        return int(np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val))
    else:
        return int(np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)[0])


def last_nonzero(arr, axis, invalid_val=-1):
    import numpy as np
    arr = arr[np.logical_not(np.isnan(arr))]
    mask = arr != 0

    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return int(np.where(mask.any(axis=axis), val, invalid_val))


def split_imu_df_by_sensor(imuDF):
    """
    Split a Pandas df containing IMU data into the individual sensors.
    i.e. left, right, chest and pocket.
    :param df: Pandas dataframe containing 9-axis IMU data.
    :return: Four n x 9 dataframes containing left, right, chest and pocket data.
    """
    import re
    dfL = imuDF[
        ["AccXlear", "AccYlear", "AccZlear", "GyroXlear", "GyroYlear", "GyroZlear", "MagXlear", "MagYlear",
         "MagZlear"]]
    dfR = imuDF[
        ["AccXrear", "AccYrear", "AccZrear", "GyroXrear", "GyroYrear", "GyroZrear", "MagXrear", "MagYrear",
         "MagZrear"]]
    dfC = imuDF[
        ["AccXchest", "AccYchest", "AccZchest", "GyroXchest", "GyroYchest", "GyroZchest", "MagXchest",
         "MagYchest",
         "MagZchest"]]
    dfP = imuDF[
        ["AccXpocket", "AccYpocket", "AccZpocket", "GyroXpocket", "GyroYpocket", "GyroZpocket",
         "MagXpocket",
         "MagYpocket",
         "MagZpocket"]]
    for df in [dfL, dfR, dfC, dfP]:
        df.columns = re.findall('[A-Z][a-z]+[A-Z]', ",".join(df.columns))
    return dfL, dfR, dfC, dfP
