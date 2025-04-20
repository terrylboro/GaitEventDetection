def get_generic_trial_nums(subjectNum : int, activitiesList : list):
    """
    Return trial numbers corresponding to given list of activities for a given subject.
    :param subjectNum:
    :param activitiesList: List of desired activities e.g. ["ShoeBox", "Walk"]
    :return: Flat list of all trial numbers for given activities.
    """
    import pandas as pd
    # load the activity info
    activityDF = pd.read_csv('../utils/info/ActivitiesIndex.csv')
    subjectDF = activityDF[activityDF["SubjectNum"] == subjectNum]
    activityTrialNums = []
    for col in subjectDF.columns:
        if not subjectDF[col].isnull().values.any() and col in activitiesList:
            trialNumsForThisActivity = [int(x) for x in subjectDF[col].values[0][1:-1].split(",")]
            activityTrialNums.extend(trialNumsForThisActivity)
    return activityTrialNums


def map_trial_num_to_activity(subjectNum : int, trialNum : int):
    """
    Return activity label for given trialNum.
    :param subjectNum:
    :param trialNum:
    :return: Flat list of all trial numbers for given activities.
    """
    import pandas as pd
    allActivitiesList = ["Walk","WalkNod","WalkShake","WalkSlow","Sit2Stand","Stand2Sit","TUG",
                         "PickUp","Reach","Static","ShoeBox","Floor2Turf","Turf2Floor"]
    for activity in allActivitiesList:
        activityNums = get_generic_trial_nums(subjectNum, [activity])
        if trialNum in activityNums:
            return activity


def get_walking_trial_nums(subjectNum):
    import pandas as pd
    # load the activity info
    activityDF = pd.read_csv('../utils/info/ActivitiesIndex.csv')
    subjectDF = activityDF[activityDF["SubjectNum"] == subjectNum]
    walkingTrialNums = []
    for col in subjectDF.columns:
        if not subjectDF[col].isnull().values.any() and col in ['Walk', 'WalkNod', 'WalkShake']:
            trialNumsForThisActivity = [int(x) for x in subjectDF[col].values[0][1:-1].split(",")]
            walkingTrialNums.extend(trialNumsForThisActivity)
    return walkingTrialNums


def get_complex_trial_nums(subjectNum):
    import pandas as pd
    # load the activity info
    activityDF = pd.read_csv('../utils/info/ActivitiesIndex.csv')
    subjectDF = activityDF[activityDF["SubjectNum"] == subjectNum]
    complexTrialNums = []
    for col in subjectDF.columns:
        if not subjectDF[col].isnull().values.any() and col in ['ShoeBox', 'Floor2Turf', 'Turf2Floor', 'WalkSlow']:
            trialNumsForThisActivity = [int(x) for x in subjectDF[col].values[0][1:-1].split(",")]
            complexTrialNums.extend(trialNumsForThisActivity)
    return complexTrialNums

def get_activity_given_trial_num(subjectNum, trialNum):
    import pandas as pd
    # load the activity info
    activityDF = pd.read_csv('../utils/info/ActivitiesIndex.csv')
    subjectDF = activityDF[activityDF["SubjectNum"] == subjectNum]


def get_subject_cycle_info(subjectNum):
    import pandas as pd
    import os
    from pathlib import Path
    thisFilePath = Path(__file__).parent.resolve()
    dfInfo = pd.read_csv(os.path.join(thisFilePath, "info/GaitCycleDetails.csv"))
    return dfInfo[dfInfo["Subject"] == subjectNum]


def get_participant_fp_strikes(subjectNum):
    import numpy as np
    subjectDF = get_subject_cycle_info(subjectNum)
    fullStrikeDF = subjectDF[np.logical_or(subjectDF.IsFullFP1Strike, subjectDF.IsFullFP2Strike)]
    leftFullStrikeDF = fullStrikeDF[fullStrikeDF.CycleSide == "Left"]
    rightFullStrikeDF = fullStrikeDF[fullStrikeDF.CycleSide == "Right"]
    return leftFullStrikeDF, rightFullStrikeDF


def get_participant_attribute(subjectNum, attributeList=[]):
    """
    Return specified background information (in a list) regarding the particpant.
    e.g. to find 18's age, enter get_participant_attribute(18, ["Age"])
    :param subjectNum: Participant number.
    :param attributeList: Information you wish to find (Sex,Age,Weight,Height,State,Reason)
    :return: df containing desired participant attributes.
    """
    import numpy as np
    import pandas as pd
    import os
    from pathlib import Path
    thisFilePath = Path(__file__).parent.resolve()
    dfInfo = pd.read_csv(os.path.join(thisFilePath, "info/ParticipantInfo.csv"), index_col="Participant")
    return dfInfo.loc[subjectNum, attributeList]


def get_non_typical_participant_nums():
    """
    Return the subjectNum of participants with movement disorders. While this could simply be the
    "State" column of ParticipantInfo.csv, this includes people who are minimally affected by their
    condition which I've decided to exclude. So in practice this just returns a hard-coded list of IDs.
    :return: List containing subjectNums of non-typical participants.
    """
    return [27, 28, 30, 31, 34, 36, 42, 43, 44, 45, 54, 55, 56, 66, 67, 68, 69, 70]

